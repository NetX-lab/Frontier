# Hardware Profiling Cookbook

A task-oriented, start-to-finish guide for collecting real hardware statistics and
turning them into a working non-dummy Frontier simulation. This complements
[the official profiling user guide](../docs/profiling/README.md) (scope/reference) with concrete commands, decision points,
and gotchas discovered while actually running this pipeline.

**Profiling on AMD ROCm (e.g. MI355X)?** See
[MI355X_ROCM_COOKBOOK.md](MI355X_ROCM_COOKBOOK.md) first — it has the
ROCm-specific environment setup and several Frontier-side workarounds
(GPU-discovery env var, missing FlashInfer backend, a vLLM API-drift issue)
that aren't covered below.

## 0. Decide what you actually need

| You want to... | You need |
|---|---|
| Compare TTFT/TPOT/E2E across parallelism/topology on **existing** devices (a100/h100/...) | Nothing new — already-checked-in CSVs under `data/profiling/compute/` |
| Simulate a **new device** you have physical access to | Section 2 (compute profiling) on that machine |
| Simulate **cross-GPU communication** realistically (TP AllReduce, PP send/recv) on a new device/node topology | Section 3 (network/collective profiling) |
| Simulate a **model not in the registry** (e.g. DeepSeek, Grok-1) | Section 4 (model config) — usually combined with 2/3 |
| Simulate hardware you **don't have physical access to** | None of the above will give you real fidelity — see [the earlier discussion in this conversation]; your options are dummy mode (no fidelity) or building a roofline predictor from vendor specs |

Two independent data domains feed two independent training/consumption paths:

```
data/profiling/compute/<device>/<model>/{linear_op,attention,moe}.csv
    → frontier/training/{linear_op,attention,moe}_trainer.py
    → cache/<model>_<hash>.pkl
    → consumed by: --execution_time_predictor_config_type {random_forrest, linear_regression}
    → drives: TTFT, TPOT (per-op compute latency)

data/profiling/network/<node_sku>/{all_reduce,send_recv}.csv
    → trained inline by vidur_cc_backend.py itself (no separate training CLI)
    → consumed by: --cc_backend_config_type vidur
    → drives: TP AllReduce cost, PP send/recv cost (cross-GPU communication)
```

Both are independent of each other — you can have real compute data with an
analytical CC backend, or real network data with dummy compute timing. Mix and
match based on what you're actually trying to measure.

## 1. Environment setup (once per machine)

```bash
conda env create -f environment_profiling.yml
conda activate frontier-profiling
python -m pip install -e ".[test]"
export PYTHONPATH=$PWD
```

This environment pulls in `vllm`, `flashinfer-python`, `torch`, `cuda-nvcc` —
deliberately excluded from the plain simulation environment. If FlashInfer JIT
compilation fails with a missing `nvcc`/`libcudart`, see the `CUDA_HOME`/
`LD_LIBRARY_PATH` fix in [AGENTS.md](../AGENTS.md#flashinfer-jit-and-nvcc).

If your hardware isn't NVIDIA/CUDA (e.g. AMD), this environment file won't work
as-is — there's no ROCm-based profiling environment checked into this repo today.
You'd need to build an equivalent HIP/ROCm environment yourself before Section 2
applies; the profiling *code* (linear_op/attention/moe wrappers) is written
against `torch`/`flashinfer`, so how much of it works unmodified on ROCm is
untested here.

## 2. Register your device

Skip this if you're profiling an already-known device (`a40`, `a100`, `a800`,
`h100`, `h800`, `h200`, `rtx_pro_6000`).

Add to `frontier/types/device_sku_type.py`:
```python
class DeviceSKUType(BaseIntEnum):
    ...
    MY_DEVICE = 9
```

Add to `frontier/config/device_sku_config.py`:
```python
@dataclass
class MyDeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = <vendor spec — only used for MFU reporting, not latency>
    total_memory_gb: int = <real HBM capacity — used for KV-cache planning>

    @staticmethod
    def get_type():
        return DeviceSKUType.MY_DEVICE
```

If you're also going to profile network collectives (Section 3), add a matching
`NodeSKUType`/`NodeSKUConfig` entry too (`frontier/types/node_sku_type.py`,
`frontier/config/node_sku_config.py`) describing how many of your devices share
NVLink/equivalent per physical node — follow the existing `*_PAIRWISE_NVLINK`/
`*_DGX` pattern.

## 3. Compute profiling (drives TTFT/TPOT)

Three operator categories, three wrapper scripts. Always try `--dry-run` first
to catch flag mistakes before spending GPU time.

```bash
export PYTHONPATH=$PWD

# 1. Linear ops (MLP, LayerNorm, projections, residual add) → linear_op.csv (or mlp.csv, both names appear in this repo)
DEVICE=my_device MODEL=meta-llama/Llama-2-7b-hf TP_SIZES="1 2 4 8" \
  bash examples/profiling/profile_linear_op.sh --dry-run
# drop --dry-run once the plan looks right

# 2. Attention (incl. chunked prefill) → attention.csv
DEVICE=my_device MODEL=meta-llama/Llama-2-7b-hf TP_SIZES="1 2 4 8" \
  ATTENTION_BACKEND=NO_OP \
  bash examples/profiling/profile_attention_chunked_prefill.sh --dry-run

# 3. MoE (gating, grouped-GEMM, routing) — only if profiling a MoE model → moe.csv
DEVICE=my_device MODEL=Qwen3-30B-A3B-tiny TP_SIZES="1 2" EP_SIZES="1 2" \
  bash examples/profiling/profile_moe.sh --dry-run
```

Output lands at `data/profiling/compute/<DEVICE>/<MODEL>/<op>.csv` — matches
exactly what `--replica_config_device`/`--replica_config_model_name` will look
for later.

**Gotchas that actually matter, confirmed from `frontier/profiling/README.md`
and the scripts themselves:**

- **`--profile_method`**: defaults to `cuda_event` in these wrapper scripts —
  good, that's what standard (non-CUDA-graph) predictor training needs. The
  *other* value, `record_function`, only feeds decode-CUDA-graph kernel-only
  training — don't mix them up if you hand-invoke `frontier.profiling.*.main`
  directly instead of using the wrappers (the wrappers already default correctly).
- **MoE uniform-routing must match end to end**: if you plan to simulate with
  `--replica_config_moe_routing_mode uniform_legacy`/`uniform_random`, you must
  profile with the matching `--routing_runtime_path uniform_topk` (not the
  default `standard_fused_topk`). Mixing standard-routing profiling data with a
  uniform-routing simulation config is explicitly called a path-mismatch bug in
  the README, not a valid approximation.
- **`--attention_backend NO_OP`** in the attention script is deliberate for
  chunked-prefill grid-search profiling — it isolates the scheduling-shape cost
  from a specific kernel backend's absolute numbers; use `FLASHINFER` if you
  want backend-specific real kernel timings instead.
- **Single-GPU profiling, TP simulated via weight sharding**: you do *not* need
  an actual multi-GPU box to generate TP=2/4/8 data points for compute
  profiling — it profiles per-shard cost on one GPU. (Network profiling in
  Section 3 is the opposite — it genuinely needs multiple real GPUs/nodes.)

**These three scripts only get you the baseline (non-mixed) attention/linear_op/moe
grids.** If your simulation needs to construct genuinely mixed prefill+decode
batches (real chunked-prefill serving under concurrency), or you're profiling
MLA attention and want real AITER kernel timings instead of the portable
`TORCH_SDPA_MLA` reference backend, those are separate, already-scripted
follow-on recipes — see [GPTOSS_TRUE_MIXED_BATCH_PROFILING.md](GPTOSS_TRUE_MIXED_BATCH_PROFILING.md)
and [AITER_KERNELS.md](AITER_KERNELS.md), and `scripts/profile_true_mixed_batch.sh` /
`scripts/profile_deepseek_aiter_mla.sh` in this same folder.

## 4. Network / collective profiling (drives TP AllReduce & PP send/recv cost)

No polished release wrapper exists for this yet (unlike Section 3's three
scripts) — invoke the profiler module directly. This one **does** need real
multi-GPU (and Ray):

```bash
export PYTHONPATH=$PWD
ray start --head   # if not already running

python -m frontier.profiling.collectives.main \
  --collective all_reduce \
  --num_workers_per_node_combinations 1 2 4 8 \
  --max_collective_size $((4096 * 8192)) \
  --precision FP16 \
  --output_dir profiling_outputs

python -m frontier.profiling.collectives.main \
  --collective send_recv \
  --num_workers_per_node_combinations 1 2 4 8 \
  --output_dir profiling_outputs
```

Output goes to `profiling_outputs/collective/<timestamp>/` — **you have to move
it into the taxonomy yourself**:
```bash
mkdir -p data/profiling/network/my_node_sku
cp profiling_outputs/collective/<timestamp>/all_reduce.csv data/profiling/network/my_node_sku/
cp profiling_outputs/collective/<timestamp>/send_recv.csv  data/profiling/network/my_node_sku/
```
`my_node_sku` must match whatever `--replica_config_network_device` you'll pass
at simulation time (e.g. `a100_pairwise_nvlink`) — confirmed from the Vidur CC
backend's own default path template:
`{profiling_data_dir}/{NETWORK_DEVICE}/all_reduce.csv`.

No separate training step needed here — `vidur_cc_backend.py` loads and fits a
model from these CSVs **inline, on first use**, cached under
`--vidur_cc_backend_config_cache_dir` (default `cache`) unless
`--vidur_cc_backend_config_no_cache` is set.

## 5. Train the compute predictors

Happens automatically the first time you run a non-dummy simulation against a
`(device, model)` pair with no cached model yet — but you can pre-train
explicitly (useful to catch data problems before a long simulation run):

```bash
python -m frontier.training.cli linear_op \
  --dataset_path data/profiling/compute/my_device/meta-llama/Llama-2-7b-hf/linear_op.csv \
  --output_dir cache --model_name meta-llama/Llama-2-7b-hf \
  --device my_device --tensor_parallel_size 1

python -m frontier.training.cli attention \
  --layer_dataset_path data/profiling/compute/my_device/meta-llama/Llama-2-7b-hf/attention.csv \
  --compute_dataset_path data/profiling/compute/my_device/meta-llama/Llama-2-7b-hf/linear_op.csv \
  --output_dir cache --model_name meta-llama/Llama-2-7b-hf \
  --device my_device --tensor_parallel_size 1

# MoE models only:
python -m frontier.training.cli moe \
  --dataset_path data/profiling/compute/my_device/Qwen3-30B-A3B-tiny/moe.csv \
  --output_dir cache --model_name Qwen3-30B-A3B-tiny \
  --device my_device --moe_tensor_parallel_size 1 --expert_parallel_size 1
```

## 6. Run the real (non-dummy) simulation

```bash
export PYTHONPATH=$PWD
python -m frontier.main \
  --simulation_mode offline --sys_arch co-location \
  --cluster_config_num_replicas 1 \
  --replica_config_device my_device \
  --replica_config_network_device my_node_sku \
  --replica_config_model_name meta-llama/Llama-2-7b-hf \
  --replica_scheduler_config_type vllm_v1 \
  --cc_backend_config_type vidur \
  --request_generator_config_type synthetic \
  --synthetic_request_generator_config_num_requests 32 \
  --length_generator_config_type fixed \
  --fixed_request_length_generator_config_prefill_tokens 512 \
  --fixed_request_length_generator_config_decode_tokens 128 \
  --fixed_request_length_generator_config_max_tokens 1024 \
  --interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps 1.0 \
  --metrics_config_output_dir outputs/real_hw_test \
  --metrics_config_run_id my_device_run
  # note: no --random_forrest_execution_time_predictor_config_enable_dummy_mode this time
```

Sanity-check the result before trusting it for anything comparative:
- `request_metrics.csv`'s `ttft`/`tpot` should be **plausible real numbers**
  (single-digit-to-low-tens of ms for a 7B-class model), not the suspiciously
  round numbers dummy mode produces (recall from this session: dummy mode gave
  us values like exactly `358.0`, `456.0` ms — a real profiled run won't line up
  on round numbers like that).
- Check `system_metrics.json`'s MFU figures aren't nonsensical (>100%, or ~0%)
  — that's usually a sign `fp16_tflops` was set wrong for the device, or the
  predicted time and the device's real per-layer FLOPs don't correspond
  (MFU is *reporting-only*, so it won't block a run, but it's a good cross-check).

## 7. Adding a model that isn't in the registry

Only needed if the model itself (not just the device) is new to Frontier. Add a
dataclass to `frontier/config/model_config.py` following the existing
`Llama2ModelConfig`/`QwenModelConfig` pattern — num_layers, embedding_dim,
num_heads, MoE-or-not, `max_position_embeddings`, etc.

Tractability depends entirely on whether the architecture is actually published:
- **Open-architecture models** (Llama, Qwen, DeepSeek, **Grok-1** — xAI
  open-sourced Grok-1's weights and architecture in March 2024: 314B params,
  MoE, 8 experts top-2) — straightforward, same pattern as existing entries.
  DeepSeek in particular reuses machinery Frontier already has: MLA (Multi-head
  Latent Attention) is already a first-class attention family
  (`LATENT_MLA_ATTENTION_FAMILY` in `frontier/attention/families.py`), with
  dedicated ledger fields (`attn_mla_decode_time`, `attn_mla_kv_up_proj_time`,
  etc.) already wired through execution-time prediction — you're mostly just
  supplying DeepSeek's real dimensions, not building new mechanism. There's
  also an existing validation asset at `tests/analysis/mla_deepseek_v2/` worth
  reading before writing a new config from scratch.
- **Closed-architecture models** (GPT-3.5/4/4o/5, and — unlike Grok-1 — later
  Grok-2/3/4, whose architectures xAI hasn't published) — no known specs to
  build from. Any config would rest on public estimates/rumors rather than
  confirmed numbers; treat comparisons involving these as illustrative, not
  quantitatively reliable, and say so if you publish results built on them.

## Quick-reference command sheet

```bash
# Compute profiling (per device, per model)
bash examples/profiling/profile_linear_op.sh --dry-run
bash examples/profiling/profile_attention_chunked_prefill.sh --dry-run
bash examples/profiling/profile_moe.sh --dry-run          # MoE models only

# Network profiling (per node SKU, needs real multi-GPU + Ray)
python -m frontier.profiling.collectives.main --collective all_reduce ...
python -m frontier.profiling.collectives.main --collective send_recv ...

# Pre-train compute predictors (optional — else trained on-demand)
python -m frontier.training.cli {linear_op|attention|moe} --dataset_path ... --device ...

# Run for real
python -m frontier.main --replica_config_device my_device --cc_backend_config_type vidur ...
# (omit --random_forrest_execution_time_predictor_config_enable_dummy_mode)

# Follow-on recipes (true-mixed-batch attention, MLA/AITER) — see scripts/ in this folder
./scripts/profile_true_mixed_batch.sh --dry-run
./scripts/profile_deepseek_aiter_mla.sh --dry-run
```
