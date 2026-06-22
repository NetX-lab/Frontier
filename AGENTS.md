# Frontier: LLM Inference Simulator (Co-location + PDD Release)

## Release Status: `pre-release-v0.2`

- Current public branch supports `co-location` and sequential PDD / `pd-disaggregation`.
- AFD / `pd-af-disaggregation` remains outside this public release surface.
- The co-location example suite uses `--cc_backend_config_type analytical` for one-click smoke runs without the optional network simulator; direct CLI experiments may still select `astra_sim_analytical` explicitly.
- The PDD example suite also uses `--cc_backend_config_type analytical` and `--no-enable_parallel_clusters` for one-click sequential smoke runs.
- collective_sim is optional. Initialize and build its submodule only when you explicitly select `--cc_backend_config_type collective_sim`.

Frontier is a modular **discrete-event simulator (DES)** for large language model (LLM) inference. This `pre-release-v0.2` branch supports the **co-location** architecture, where prefill and decode run in a single monolithic cluster, and sequential **PDD / `pd-disaggregation`**, where prefill and decode run in separate clusters with KV cache transfer between them.

The supported PDD path requires sequential cluster execution. If a user selects `pd-disaggregation` with parallel clusters enabled, Frontier fails fast with this message:

```text
Error: pd-disaggregation public release support requires --no-enable_parallel_clusters. Parallel cluster processing for pd-disaggregation is not included in this release.
```

If a user selects `pd-af-disaggregation` or other unsupported disaggregated research surfaces, Frontier still fails fast with the guarded disaggregation release error.

This AGENTS.md release guide is intended to be the **authoritative entry point** for users and developers. Older documents in the repo may contain deeper narrative explanations but may lag behind the current code.

## Contents

- [What Frontier Simulates](#what-frontier-simulates)
- [Key Features](#key-features)
- [Supported System Architectures & Mode Compatibility](#supported-system-architectures--mode-compatibility)
- [Repository Layout](#repository-layout)
- [User Guides](#user-guides)
- [Install / Environment](#install--environment)
- [Docker Environment](#docker-environment)
- [Quick Start: Run a Simulation](#quick-start-run-a-simulation)
- [Examples](#examples)
- [Configuration Model](#configuration-model)
- [System Architecture](#system-architecture)
- [Metrics & Outputs](#metrics--outputs)
- [Canonical TTFT Contract for Frontier vs vLLM V1 Online Alignment](#canonical-ttft-contract-for-frontier-vs-vllm-v1-online-alignment)
- [Training (Execution-Time & Network Models)](#training-execution-time--network-models)
- [Profiling Utilities](#profiling-utilities)
- [Tests](#tests)
- [Contributing](#contributing)
- [License](#license)
- [Other Documentation](#other-documentation)

## What Frontier Simulates

Frontier models an LLM serving system as a set of clusters and replicas processing incoming requests over time. It uses a DES event loop to represent:

- **Request arrival** and workload generation (synthetic or trace-based)
- **Hierarchical scheduling** (global → cluster → replica → pipeline stage)
- **Execution time prediction** for model operations (ML-driven or “dummy mode”)
- **MoE execution modeling**, including Expert Parallelism (EP) synchronization and token imbalance
- **Speculative decoding / MTP runtime modeling** for supported `spec_decode` methods
- **Prefix caching** with block-hash-based KV reuse on supported schedulers

## Key Features

- `co-location` and sequential `pd-disaggregation` system architectures for this release
- Runtime guard for `pd-af-disaggregation`, parallel PDD clusters, and unsupported disaggregated research surfaces
- MoE support (EP synchronization, routing and imbalance modeling)
- Speculative decoding support via `frontier/spec_decode/` and `ReplicaConfig.speculative_decoding_config`
- Prefix caching for supported replica schedulers (`vllm_v1`, `sglang`)
- Pluggable **communication-cost backend**:
  - ASTRA-Sim-inspired analytical backend (default for public examples and direct CLI defaults)
  - Collective-sim topology-aware backend (optional; requires explicit `--cc_backend_config_type collective_sim`)
  - Analytical backend
  - Vidur (sklearn-based) backend trained on profiling data
- Detailed metrics collection + optional plots
- Optional per-cluster **event logging** for debugging

## Supported System Architectures & Mode Compatibility

This release supports two runtime architectures:

- `co-location`: Monolithic mode with a single cluster.
- `pd-disaggregation`: Sequential PDD mode with separate `PREFILL` and unified `DECODE` clusters. Public examples use `--no-enable_parallel_clusters`.

The CLI/config parser still accepts the historical `sys_arch` choices so existing parameter parsing structures remain stable. Runtime behavior is stricter:

- `offline + co-location` is supported.
- `online + co-location` is supported where the selected scheduler/runtime path supports online mode.
- `offline + pd-disaggregation` is supported for the public PDD scripts.
- `online + pd-disaggregation` is supported for the public PDD scripts.
- `pd-af-disaggregation` aborts during `SimulationConfig.__post_init__()` with the guarded disaggregation release error.
- `pd-disaggregation` aborts unless `--no-enable_parallel_clusters` is provided.

Important runtime constraints:

- `sglang` is available only for `co-location` / `MONOLITHIC`.
- `decode_cuda_graph_mode` is intended for `co-location` and `pd-disaggregation`.
- `use_cuda_graph=True` is not part of this release because the guarded PD+AF path previously owned that setting.
- When speculative decoding is enabled, Frontier currently requires `decode_cuda_graph_mode='none'` unless the diagnostic opt-in is explicitly enabled.

## Repository Layout

Top-level directories you will commonly use:

- `frontier/`: simulator source code (the Python package)
- `data/`: profiling datasets and model config assets consumed by predictors/backends
- `docs/`: user guides for CLI usage, profiling, and predictor training
- `examples/`: runnable shell scripts demonstrating various features (see [Examples](#examples))
- `tests/`: many runnable scripts (bash + python) used as integration/validation harnesses
- `frontier/cc_backend/backends/collective-sim/`: optional backend submodule, required only for explicit `collective_sim` runs
- `cache/`: generated sklearn model caches and predictor caches when training / caching is enabled

Core `frontier/` subpackages (high-level):

- `frontier/config/`: dataclass-based configuration system + CLI flattening
- `frontier/entities/`: `Request`, `Batch`, `Cluster`, `Replica`, etc.
- `frontier/events/`: DES events that drive the simulation
- `frontier/scheduler/`: hierarchical schedulers (global/cluster/replica/stage)
- `frontier/execution_time_predictor/`: sklearn-based latency predictors and MoE support
- `frontier/cc_backend/`: communication-cost backend (`collective_sim`, `astra_sim_analytical`, `analytical`, `vidur`)
- `frontier/metrics/`: metrics store, plots, traces
- `frontier/training/`: model training CLI for predictors
- `frontier/profiling/`: utilities for collecting profiling data
- `frontier/spec_decode/`: speculative decoding / MTP helpers and runtime contracts

Entry points:

- `python -m frontier.main`: run a simulation
- `python -m frontier.training`: train prediction models
- `python -m frontier.training.cli`: train prediction models

## User Guides

Use this README for the project overview and installation path. Use the focused guides below when running a workflow:

| Guide                      | Use                                                                                                                           |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `docs/cli/README.md`       | CLI environment, co-location examples, direct `frontier.main` flags, metrics output, and guarded release paths.               |
| `docs/profiling/README.md` | Public profiling wrappers, operator coverage, Chunked Prefill attention profiling, output taxonomy, and simulator CSV smokes. |
| `docs/training/README.md`  | Standalone predictor training, cache management, and E2E on-demand predictor training behavior.                               |

The release examples remain under `examples/`. The `docs/` guides explain how to adapt those examples into custom commands without starting from the full dataclass-generated CLI surface.

## Install / Environment

### Python

Frontier is a Python project with a minimal top-level `pyproject.toml` for the co-location release surface.

For conda users, start from the checked-in minimal release environment:

```bash
conda env create -f environment.yml
conda activate frontier
```

For local development, install the package in editable mode or run scripts with `PYTHONPATH=$PWD` from the repo root.

```bash
python -m pip install -e ".[test]"
export PYTHONPATH=$PWD
```

### Key dependencies

A minimal environment that can import and run the simulator typically needs:

- `numpy`, `pandas`, `scipy`
- `scikit-learn` (sklearn-based predictors/backends)
- `plotly` (metrics/plotting; currently imported by default)
- `fasteners` and `ddsketch` (predictor cache locking and metric CDF sketches)
- `pytest` for the release sanity tests

Common optional dependencies depending on what you run:

- `wandb` (optional; required only when W&B logging is explicitly configured)
- `matplotlib` (some analysis/visualization scripts)
- `torch` and GPU-specific packages for real GPU profiling runs

The minimal `environment.yml` intentionally does not include large GPU packages such as `torch`, `vllm`, `flashinfer-python`, or `triton`. Those are profiling/alignment dependencies and should be installed only in a dedicated GPU profiling environment.

### Optional collective-sim backend build

The co-location example suite uses `--cc_backend_config_type analytical` and does not require the `collective_sim` binary. Direct CLI experiments may explicitly select `astra_sim_analytical`; initialize and build the `collective_sim` submodule only when you explicitly select `--cc_backend_config_type collective_sim`:

```bash
git submodule update --init --recursive frontier/cc_backend/backends/collective-sim
cd frontier/cc_backend/backends/collective-sim/sim
make -j"$(nproc)"
```

If the binary exists but was built on a different host or against an incompatible `GLIBC`, rebuild it in the current runtime:

```bash
cd frontier/cc_backend/backends/collective-sim/sim
make -B -j"$(nproc)"
```

### Dedicated GPU profiling environment

The simulator can run from the minimal `environment.yml`, but real GPU profiling needs heavier dependencies. Use the dedicated profiling file when collecting operator timing data:

```bash
conda env create -f environment_profiling.yml
conda activate frontier-profiling
python -m pip install -e ".[test]"
export PYTHONPATH=$PWD
```

`environment_profiling.yml` intentionally includes `vllm`, `flashinfer-python`, `torch`, `triton`, and `cuda-nvcc`. If you already have an existing environment with `vllm` and `flashinfer` configured, you can use that existing environment for profiling instead of creating a new one; make sure `PYTHONPATH` points at this repository and run the profiling entry points from the repo root.

### FlashInfer JIT and `nvcc`

FlashInfer JIT compilation requires a working `nvcc` and linkable CUDA runtime libraries. The profiling environment installs `cuda-nvcc`, `cuda-cudart`, and `cuda-cudart-dev` from conda-forge so FlashInfer can compile and link kernels inside the Conda environment. If profiling fails with an error that mentions missing CUDA compiler tools, first check `CUDA_HOME`; a stale host path such as `/usr/local/cuda-*` can override the Conda toolchain. If the linker fails with `cannot find -lcudart`, make the Conda runtime library directory visible to the compiler and dynamic linker:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Recommended verification:

```bash
conda run -n frontier-profiling bash -c '
  export CUDA_HOME="$CONDA_PREFIX"
  export LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  which nvcc
  nvcc --version
  python - <<"PY"
from torch.utils.cpp_extension import CUDA_HOME
import torch
import flashinfer
print("torch_cuda", torch.version.cuda)
print("cpp_extension_CUDA_HOME", CUDA_HOME)
print("flashinfer", flashinfer.__version__)
PY
'
```

## Docker Environment

A pre-built image is available for release validation and for users who want an image-specific environment rather than local Conda setup. Pull the image first:

```bash
docker pull fengyicheng/frontier-env
```

Run a container by mounting the local repository into `/workspace/frontier`:

```bash
docker run --rm --gpus all \
  --shm-size 16g \
  -v "$PWD":/workspace/frontier:ro \
  --tmpfs /workspace/frontier/outputs:mode=775 \
  --tmpfs /workspace/frontier/cache:mode=775 \
  -w /workspace/frontier \
  -e PYTHONPATH=/workspace/frontier \
  -e PYTHONDONTWRITEBYTECODE=1 \
  fengyicheng/frontier-env \
  bash -lc '
    if [ -z "${FRONTIER_DOCKER_PYTHON:-}" ]; then
      FRONTIER_DOCKER_PYTHON="$(find / -path "*/envs/vidur_te/bin/python" \( -type f -o -type l \) 2>/dev/null | sort | head -n 1)"
    fi
    if [ -z "$FRONTIER_DOCKER_PYTHON" ]; then
      echo "Python executable not found inside the container. Set FRONTIER_DOCKER_PYTHON=/path/to/python." >&2
      exit 1
    fi
    "$FRONTIER_DOCKER_PYTHON" -c "import pytest"
    "$FRONTIER_DOCKER_PYTHON" -m pytest tests/unit/test_open_source_release_arch_guard.py -q -p no:cacheprovider
  '
```

`FRONTIER_DOCKER_PYTHON` is the image-specific Python executable. The command above discovers the `vidur_te` environment automatically inside `fengyicheng/frontier-env`; if you build or use a different image, replace this path discovery by setting `FRONTIER_DOCKER_PYTHON` to the Python executable inside that image. The `--tmpfs` mounts keep generated `outputs` and `cache` files out of the read-only source mount during smoke tests.

### Docker troubleshooting and common pitfalls

- **NVIDIA Container Toolkit**: GPU access requires the NVIDIA Container Toolkit on the host. If `--gpus all` fails, verify the toolkit installation and run `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi` or an equivalent CUDA base image test.
- **Shared memory**: Heavy GPU workloads, multiprocessing, and profiling can need more shared memory than Docker's default. Keep `--shm-size 16g` or increase it for multi-GPU runs.
- **driver compatibility**: The host NVIDIA driver must be compatible with the CUDA runtime in the container. If imports work but CUDA initialization fails, compare host `nvidia-smi` output with the image CUDA version.
- **Mounted repository permissions**: The recommended command mounts the repo read-only and uses `--tmpfs /workspace/frontier/outputs` plus `--tmpfs /workspace/frontier/cache` for generated files. Remove `:ro` only when intentionally editing inside the container.

## Quick Start: Run a Simulation

Because the CLI is generated from dataclasses, the flag set is large. The most reliable workflow is:

1. Start from an existing script in `examples/` or `tests/`.
2. Replace only the parameters you need.

### Example: Dense co-location using dummy execution time

The co-location dense examples are intended to be runnable without profiling data. The example suite defaults to `--cc_backend_config_type analytical`, so it does not require the optional `collective_sim` binary. If you explicitly select `--cc_backend_config_type collective_sim`, build the collective-sim binary first and verify `frontier/cc_backend/backends/collective-sim/sim/datacenter/htsim_ndp` exists before running:

```bash
# From repo root
export PYTHONPATH=$PWD
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1

bash examples/architecture/co-location/offline/dense_model_basic.sh
```

### Example: MoE co-location

Use the MoE co-location wrappers for shared-domain MoE smoke runs. These wrappers default to `--cc_backend_config_type analytical`; `collective_sim` remains an explicit opt-in backend:

```bash
export PYTHONPATH=$PWD
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1

bash examples/architecture/co-location/offline/moe_model_basic.sh
```

### Example: Dense PDD

The PDD dense example is also runnable without profiling data. It uses sequential PDD through `--no-enable_parallel_clusters` and writes the same CSV/JSON metrics artifacts as the co-location examples:

```bash
export PYTHONPATH=$PWD
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1

bash examples/architecture/pdd/offline/dense_model_basic.sh
```

## Examples

The release-supported example surface is split between runtime architecture recipes and profiling recipes:

```text
examples/
├── README.md
├── fixtures/
│   └── prefix_cache_shared_session_trace.csv
├── architecture/
│   ├── README.md
│   ├── pdd/
│   │   ├── run_all.sh
│   │   ├── dense_model_basic.sh
│   │   ├── offline/
│   │   │   ├── dense_model_basic.sh
│   │   │   ├── moe_model_basic.sh
│   │   │   ├── thinking_mode_basic.sh
│   │   │   ├── moe_spec_dec.sh
│   │   │   └── moe_prefix_caching.sh
│   │   └── online/
│   │       ├── dense_model_basic_online.sh
│   │       ├── moe_model_basic_online.sh
│   │       ├── thinking_mode_basic_online.sh
│   │       ├── moe_spec_dec_online.sh
│   │       └── moe_prefix_caching_online.sh
│   └── co-location/
│       ├── run_all.sh
│       ├── offline/
│       │   ├── dense_model_basic.sh
│       │   ├── moe_model_basic.sh
│       │   ├── thinking_mode_basic.sh
│       │   ├── moe_spec_dec.sh
│       │   └── moe_prefix_caching.sh
│       └── online/
│           ├── dense_model_basic_online.sh
│           ├── moe_model_basic_online.sh
│           ├── thinking_mode_basic_online.sh
│           ├── moe_spec_dec_online.sh
│           └── moe_prefix_caching_online.sh
└── profiling/
    ├── README.md
    ├── profile_linear_op.sh
    ├── profile_attention_chunked_prefill.sh
    ├── profile_moe.sh
    ├── smoke_metadata.sh
    ├── smoke_simulator_dense_csv.sh
    └── smoke_simulator_moe_csv.sh
```

PDD recipes:

| Script                                                        | Purpose                                  | Default runtime behavior                                                                                       |
| ------------------------------------------------------------- | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `examples/architecture/pdd/run_all.sh`                        | Full PDD suite                           | Runs all five offline PDD cases and all five online PDD cases                                                  |
| `examples/architecture/pdd/offline/dense_model_basic.sh`      | Offline dense PDD baseline               | Sequential `pd-disaggregation`, analytical backend, Chunked Prefill enabled, CSV/JSON metrics enabled          |
| `examples/architecture/pdd/offline/moe_model_basic.sh`        | Offline MoE PDD baseline                 | Sequential `pd-disaggregation`, shared-domain MoE invariant, Chunked Prefill enabled, CSV/JSON metrics enabled |
| `examples/architecture/pdd/offline/thinking_mode_basic.sh`    | Offline Thinking Mode PDD smoke          | Thinking Mode enabled and records prefill-to-decode KV transfer handoffs                                       |
| `examples/architecture/pdd/offline/moe_spec_dec.sh`           | Offline MoE PDD Speculative Decoding/MTP | Speculative Decoding / MTP enabled, `decode_cuda_graph_mode=none` to avoid the current runtime conflict        |
| `examples/architecture/pdd/offline/moe_prefix_caching.sh`     | Offline MoE PDD Prefix Caching recipe    | Prefix Caching enabled against `examples/fixtures/prefix_cache_shared_session_trace.csv`                       |
| `examples/architecture/pdd/online/dense_model_basic_online.sh` | Online dense PDD baseline                | Mirrors the dense offline case with `--simulation_mode online`                                                 |
| `examples/architecture/pdd/online/moe_model_basic_online.sh`  | Online MoE PDD baseline                  | Mirrors the MoE offline case with `--simulation_mode online`                                                   |

Co-location recipes:

| Script                                                                   | Purpose                                       | Default runtime behavior                                                                                                            |
| ------------------------------------------------------------------------ | --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `examples/architecture/co-location/run_all.sh`                           | Full co-location suite                        | Runs all five offline cases and all five online cases                                                                               |
| `examples/architecture/co-location/offline/dense_model_basic.sh`         | Offline dense co-location baseline            | `--cc_backend_config_type analytical`, `decode_cuda_graph_mode=full_decode_only`, Chunked Prefill enabled, CSV/JSON metrics enabled |
| `examples/architecture/co-location/offline/moe_model_basic.sh`           | Offline MoE co-location baseline              | `--cc_backend_config_type analytical`, shared-domain MoE invariant, Chunked Prefill enabled, CSV/JSON metrics enabled               |
| `examples/architecture/co-location/offline/thinking_mode_basic.sh`       | Offline Thinking Mode co-location smoke       | `--cc_backend_config_type analytical`, Thinking Mode enabled, CSV/JSON metrics enabled                                              |
| `examples/architecture/co-location/offline/moe_spec_dec.sh`              | Offline MoE Speculative Decoding / MTP recipe | Speculative Decoding / MTP enabled, `decode_cuda_graph_mode=none` to avoid the current runtime conflict                             |
| `examples/architecture/co-location/offline/moe_prefix_caching.sh`        | Offline MoE Prefix Caching recipe             | Prefix Caching enabled against `examples/fixtures/prefix_cache_shared_session_trace.csv`                                            |
| `examples/architecture/co-location/online/dense_model_basic_online.sh`   | Online dense co-location baseline             | Mirrors the dense offline case with `--simulation_mode online`                                                                      |
| `examples/architecture/co-location/online/moe_model_basic_online.sh`     | Online MoE co-location baseline               | Mirrors the MoE offline case with `--simulation_mode online`                                                                        |
| `examples/architecture/co-location/online/thinking_mode_basic_online.sh` | Online Thinking Mode co-location smoke        | Mirrors the Thinking Mode offline case with `--simulation_mode online`                                                              |
| `examples/architecture/co-location/online/moe_spec_dec_online.sh`        | Online MoE Speculative Decoding / MTP recipe  | Mirrors the speculative decoding offline case with `--simulation_mode online`                                                       |
| `examples/architecture/co-location/online/moe_prefix_caching_online.sh`  | Online MoE Prefix Caching recipe              | Replays the same prefix-cache fixture with `--simulation_mode online`                                                               |

Quick start with examples:

```bash
# Basic PDD example
bash examples/architecture/pdd/offline/dense_model_basic.sh

# Full PDD suite
bash examples/architecture/pdd/run_all.sh

# Basic co-location (monolithic) MoE example
bash examples/architecture/co-location/offline/moe_model_basic.sh

# Advanced MoE recipes
bash examples/architecture/co-location/offline/moe_spec_dec.sh
bash examples/architecture/co-location/offline/moe_prefix_caching.sh
```

The dense, MoE, Thinking Mode, Speculative Decoding / MTP, and Prefix Caching examples all default to `--cc_backend_config_type analytical`. `collective_sim` is optional: build `frontier/cc_backend/backends/collective-sim/sim/datacenter/htsim_ndp` only when you explicitly select `--cc_backend_config_type collective_sim`.

Dummy mode (`--random_forrest_execution_time_predictor_config_enable_dummy_mode`) skips ML predictor training and profiling metadata loading; it is suitable for smoke tests, not realistic latency prediction. For production simulations, disable dummy mode and provide matching CSV datasets under `data/profiling/compute/<device>/<model>/`.

Profiling examples cover three operator classes and write to the canonical compute taxonomy:

```bash
bash examples/profiling/profile_linear_op.sh --dry-run
bash examples/profiling/profile_attention_chunked_prefill.sh --dry-run
bash examples/profiling/profile_moe.sh --dry-run
bash examples/profiling/smoke_simulator_dense_csv.sh
bash examples/profiling/smoke_simulator_moe_csv.sh
```

PNG plot export is optional and requires `kaleido`. If `kaleido` is not installed, Plotly may warn about image export, but CSV/JSON metrics are still produced.

See `examples/README.md`, `examples/architecture/README.md`, and `examples/profiling/README.md` for full documentation.

## Configuration Model

Frontier uses nested dataclasses for configuration (see `frontier/config/config.py`). CLI parsing is implemented by flattening the nested dataclasses into a large set of `--<field>` flags (see `frontier/config/flat_dataclass.py`).

Practical implications:

- Many flags look like `--cluster_config_<subconfig>_<field>` for co-location settings.
- Polymorphic configs use an explicit `*_type` field (e.g., request generator type).
- Defaults exist for many fields, but some options may be required depending on which config objects are instantiated.
- Backend-specific CC subconfigs use backend-prefixed flat flags such as `--analytical_cc_backend_config_*`, `--collective_sim_cc_backend_config_*`, and `--astra_sim_analytical_cc_backend_config_*`.
- PDD cluster-specific parser fields such as `--cluster_config_prefill_*` and `--cluster_config_decode_*` are release-supported for `pd-disaggregation`.
- Historical AFD parser fields such as `--cluster_config_decode_attn_*`, `--cluster_config_decode_ffn_*`, and M2N transfer fields remain guarded out of this release.
- KV-cache transfer parser fields such as `--kv_cache_transfer_config_type` and `--analytical_kv_cache_transfer_config_*` are release-supported for the public PDD path.
- For `astra_sim_analytical`, runtime-materialized layout fields such as `cluster_servers`, `cluster_gpus_per_server`, and `runtime_*` are internal only and intentionally omitted from the public CLI.

Configuration inheritance for the co-location path:

1. Prefer explicitly provided co-location fields.
2. Fall back to the base `replica_config`.
3. Fall back to dataclass defaults.

## System Architecture

Frontier uses a hierarchical scheduling and simulation approach. In this release, the supported runtime maps those layers onto a single `MONOLITHIC` cluster.

### 1. Scheduler Hierarchy

The scheduling logic is split across four distinct layers to mirror real-world serving systems:

1.  **Global Scheduler** (`BaseGlobalScheduler`):
    - **Role**: Top-level orchestrator.
    - **Entry Point**: Receives all incoming `RequestArrivalEvent`s.
    - **Routing**: Routes requests into the monolithic cluster in this release.
    - _Implementation Note_: While named `BaseGlobalScheduler`, this class is instantiated directly in `simulator.py` and serves as the concrete global scheduler.

2.  **Cluster Scheduler** (`ClusterSchedulerRegistry`):
    - **Role**: Manages workload distribution within a specific `ClusterType` (e.g., selecting which Replica gets a request).
    - **Implementations**:
      - `RoundRobinClusterScheduler`: Distributes requests cyclically.
      - `LORClusterScheduler`: Least Outstanding Requests (load balancing).
      - `RandomClusterScheduler`: Random assignment.

3.  **Replica Scheduler** (`ReplicaSchedulerRegistry`):
    - **Role**: Operates at the level of a single `Replica` (GPU node/instance).
    - **Responsibilities**: Request batching policy (e.g., continuous batching), memory/block allocation (paging), preemption, prefix-cache-aware admission, and speculative decoding metadata flow on supported schedulers.
    - **Implementations**:
      - `VLLMReplicaScheduler`: Models vLLM's scheduling logic.
      - `SarathiReplicaScheduler`: Models Sarathi-serve (chunked prefill).
      - `OrcaReplicaScheduler`: Models Orca (iteration-level scheduling).
      - `VllmV1EngineReplicaScheduler`: Models vLLM V1 architecture.
      - `SGLangStyleReplicaScheduler`: Models SGLang-style prefill-first scheduling for monolithic runs.

4.  **Replica Stage Scheduler** (`ReplicaStageScheduler`):
    - **Role**: Manages the low-level execution pipeline stages (Tensor Parallelism, Pipeline Parallelism).
    - **Interaction**: Direct interface with the `ExecutionTimePredictor` to determine operation latencies.

### 2. Key Entities

- **Cluster**: Represents the monolithic compute pool. It manages a set of Replicas and lazy-loads the communication (`CCBackend`) model.
- **Replica**: Represents a physical serving instance (e.g., an 8-GPU node). It validates hardware configurations (TP/PP sizes) and maintains local state (memory usage, running batches).
- **Request**: Tracks the full lifecycle of an inference query.
  - Tracks latency components: Arrival, Scheduling Delay, and Preemption overhead.
- **Batch**: A logical grouping of requests executing together.
  - **Global ID**: Used to coordinate EP sub-batches during synchronization/all-gather.

### 3. Execution Time Prediction & Events

- **Predictors**: `SklearnExecutionTimePredictor` uses ML models (Random Forest/Linear Regression) trained on profiling data to predict granular operation latencies (Found in `frontier/execution_time_predictor/`).
- **CC Backend**: Predicts collective communication costs (AllReduce, AllGather). Release-supported implementations include `CollectiveSimCCBackend`, `AstraSimAnalyticalCCBackend`, `VidurCCBackend`, and `AnalyticalCCBackend`.
- **Event Logic**: The simulation is driven by specific event types:
  - `ClusterBatchEndEvent`: Handles cluster-local completion for monolithic batches.
  - `GlobalBatchEndEvent`: Handles request-level decode completion, metrics, and memory release.

## Metrics & Outputs

Metrics are collected by `frontier/metrics/metrics_store.py` and written under the configured metrics root (see `metrics_config`). Each run is normalized into:

```text
outputs/metrics/<model_type>/<workload_type>/<run_id>/
```

For example:

```text
outputs/metrics/meta_llama_llama_2_7b_hf/offline_batch/run_001/request_metrics.csv
```

Output can include:

- `request_metrics.csv`: per-request and per-token latency distributions.
- `system_metrics.json`: aggregate sections such as `simulation_metadata`, `ttft_statistics`, `tpot_statistics`, `request_e2e_time_statistics`, `throughput_metrics`, `spec_decode_statistics`, `preemption_statistics`, and `system_architecture_info`.
- Batch-level statistics when enabled.
- Optional plots (Plotly).
- Optional Chrome trace output when enabled.
- Optional `metrics_ground_truth.jsonl` request instrumentation records when `metrics_config.enable_metrics_ground_truth_trace=True`.

Latency fields such as TTFT, TPOT, and request E2E time are reported in milliseconds (`ms`). `tpot_statistics` is computed only for requests with `num_decode_tokens > 1`; the JSON note records how many requests were included. For strict throughput cross-validation, enable `metrics_ground_truth.jsonl`; `request_metrics.csv` alone does not contain every wall-clock interval needed to independently recompute duration.

Note: Metrics plotting imports `plotly` unconditionally, so `plotly` must be installed to run the main simulator. The example scripts disable plot and trace outputs by default while still writing CSV/JSON metrics.

### Canonical TTFT Contract for Frontier vs vLLM V1 Online Alignment

For the online alignment suite under `tests/comparison/chunked_prefill_online/`, the canonical `TTFT` definition is frozen to:

- `queue-visible request arrival -> request prefill completion`

This is intentionally narrower than the official streaming/client-side "first token visible" meaning. The reason is pragmatic: it avoids mixing currently unresolved request-visible tails and other output-side overhead into the primary TTFT error budget.

When reading artifacts, keep the following distinction:

- Canonical comparison TTFT:
  - Frontier: `request_metrics.csv` column `ttft`
  - vLLM: `comparison/vllm_request_metrics.csv` column `ttft_ms`, reconstructed from batch-log prefill completion
- Legacy/raw TTFT references:
  - `vllm_clean/vllm_request_metrics.csv` client-visible `ttft_ms`
  - `vllm_server_request_metrics.jsonl` field `ttft`

Those raw vLLM TTFT values are still useful for debugging and historical context, but they are not the canonical Frontier-vs-vLLM comparison target anymore.

## Training (Execution-Time & Network Models)

Frontier includes a standalone training CLI for sklearn models:

```bash
export PYTHONPATH=$PWD
python -m frontier.training -h
python -m frontier.training.cli -h
```

Typical workflows train models from CSV profiling datasets under `data/profiling/` and save artifacts into `cache/`.

Standalone training is optional for normal E2E simulation. When dummy predictor mode is disabled, Frontier checks the predictor cache through `metrics_config.cache_dir`. If a required predictor is missing, the simulator trains it during initialization from the configured profiling CSVs and writes the trained model to cache. Run the training CLI separately when you want to pre-warm cache files, debug a profiling CSV schema issue, or compare predictor artifacts without launching a full simulation.

See `docs/training/README.md` for the command-level training guide.

## Profiling Utilities

`frontier/profiling/` contains the profiling implementation. The user-facing release examples live in `examples/profiling/`, while the legacy internal helper scripts under `frontier/profiling/example/` remain available for backward reference.

The top-level examples cover `linear_op`, `attention`, and `moe` operator classes. `examples/profiling/profile_attention_chunked_prefill.sh` demonstrates attention profiling under a Chunked Prefill runtime state. `examples/profiling/smoke_simulator_dense_csv.sh` and `examples/profiling/smoke_simulator_moe_csv.sh` feed checked-in CSV profiles directly into the simulator with dummy predictor mode disabled. The MoE downstream smoke uses `uniform_random` routing because the checked-in tiny MoE CSV contains `routing_runtime_path=uniform_topk` rows.

All release-facing profiling examples use the canonical taxonomy:

```text
data/profiling/compute/<device>/<model>/
├── linear_op.csv
├── attention.csv
└── moe.csv
```

Lightweight profiling planning, schema, metadata, migration, and validation modules can be imported from the minimal environment. Real GPU profiling entry points require additional packages such as `torch`, and explicit `collective_sim` runs may require optional backend build dependencies.

See `docs/profiling/README.md` for the public profiling workflow and downstream simulator checks. For direct simulation CLI usage, see `docs/cli/README.md`.

### vLLM vs Frontier memory variable mapping (KV block initialization)

| Concept                                     | vLLM variable / formula                                                           | Frontier variable / formula                                                                                                  | Notes                                                                                               |
| ------------------------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Requested memory budget                     | `requested_memory = total_memory * gpu_memory_utilization`                        | `requested_memory = total_memory * gpu_memory_utilization`                                                                   | Same high-level budgeting idea.                                                                     |
| Weight/parameter memory used in subtraction | `weights_memory` (runtime measured model load memory delta)                       | `param_memory` (`2 * num_parameters_per_device` from `ParamCounter`)                                                         | Frontier defaults to analytical param counting; can be adjusted with runtime-measured weights path. |
| Non-weight non-KV term                      | Implicit in `non_kv_cache_memory` decomposition (`torch_peak + non_torch`)        | `non_kv_cache_overhead_bytes`                                                                                                | Frontier exposes this as an explicit input/configurable calibration term.                           |
| Full non-KV memory                          | `non_kv_cache_memory = weights_memory + torch_peak_increase + non_torch_increase` | `non_kv_cache_memory_bytes` is computed in profiling module, then typically split into `param_memory + overhead` for planner | Frontier profile result contains full value, but planner consumes split terms.                      |
| Available memory for KV cache               | `available_kv = requested_memory - non_kv_cache_memory`                           | `available_kv = requested_memory - param_memory - non_kv_cache_overhead_bytes`                                               | Formally equivalent when `param_memory` aligns with profiled `weights_memory`.                      |
| KV block count                              | `num_blocks = floor(available_kv / page_size / num_layers)`                       | `num_blocks = floor(available_kv / page_size / num_layers)`                                                                  | Same final block-count formula.                                                                     |

Practical implication: Frontier intentionally uses a split interface (`param_memory` + `overhead`) to support three modes (`memory_planner`, `memory_planner_profiled`, `explicit`) while keeping compatibility with vLLM-style memory accounting.

## Tests

The repo contains a large set of scripts in `tests/`. For this release branch, start with PDD, co-location, and release-guard coverage:

- `comm_backend_tests/`: Tests for Communication Cost (CC) backends.
- `integration/`: workflow-level tests for CUDA graph, prefix cache, and spec decode where applicable to co-location.
- `debug/`: runnable end-to-end smoke and development scripts.
- `comparison/` and `analysis/`: Frontier vs vLLM validation and RCA tooling; some historical files cover guarded architectures and should be treated as non-release references.

Start with:

- `pytest tests/unit/test_pdd_public_surface_docs.py tests/unit/test_examples_pdd_scripts.py -q`
- `bash tests/debug/e2e-level/monolith_mode/scripts/test_dense_tp2_pp2_dummy.sh`
- `bash tests/debug/e2e-level/monolith_mode/scripts/test_moe_tp2_ep2_pp2_dummy.sh`

## Contributing

- Follow the architectural boundaries: events drive state changes; schedulers should be layered (global → cluster → replica → stage).
- Prefer incremental, backward-compatible changes.
- When you add a new feature, update docs and add a runnable test scenario.

## License

Frontier `pre-release-v0.2` is released under the MIT License. See `LICENSE` for the full text.

## Other Documentation

These checked-in docs are useful follow-up references:

- `docs/cli/README.md` - CLI user guide for co-location simulation and metrics output
- `docs/profiling/README.md` - profiling user guide for public wrappers and simulator CSV smokes
- `docs/training/README.md` - predictor training guide, including E2E on-demand cache training
- `examples/README.md` - runnable example catalog
- `frontier/profiling/README.md` - profiling workflow details
- `tests/comparison/README.md` - Frontier vs vLLM comparison workflow
