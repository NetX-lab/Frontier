# ROCm and MI355X Profiling

## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-17 | Corrected TP8 GDN launch to use eight distributed processes. |
| 2026-09-17 | Clarified native mixed-phase rejection versus the temporary simulator approximation. |
| 2026-09-14 | Added the standard MI355X profiling path and the experimental SGLang boundary. |

## Scope

This page describes the GPU-side steps for collecting standard Frontier
profiles on an AMD MI355X or MI355X_UBB worker. `MI355X` is the ROCm platform
entry and `MI355X_UBB` describes the eight-GPU-per-node topology. The selected
SKU does not silently select a communication backend or rewrite a user-provided
topology.

The CPU master can prepare commands and validate schemas. It cannot establish
HIP kernel, AITER, MXFP4, RCCL, or ROCm event correctness without an MI355X
worker.

## Environment

Use the profiling environment and keep the repository on the shared workspace:

```bash
conda activate frontier-profiling
python -m pip install -e ".[test]"
export PYTHONPATH=$PWD
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1
```

On the GPU worker, verify that the visible devices and ROCm runtime match the
requested TP/EP plan before collecting data. Keep the launcher-provided
`ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, and `CUDA_VISIBLE_DEVICES`
values unchanged. The common accelerator discovery path uses that visibility
instead of reconstructing a capture manifest.

## Standard measurement family

ROCm standard timing uses `DEVICE_EVENT`. The producer rejects
`cuda_event` on ROCm so that CUDA and ROCm event rows cannot be mixed in one
predictor identity. Standard outputs use the canonical taxonomy:

```text
data/profiling/compute/<device>/<model>/
├── linear_op_device_event.csv
├── attention_device_event.csv
├── moe_device_event.csv
└── gdn.csv
```

Collect generic operators with the release wrappers:

```bash
bash examples/profiling/profile_linear_op.sh \
  --model Qwen3.8-2.4T-A95B-Quark-MXFP4 \
  --device mi355x \
  --profile-method device_event \
  --tp-sizes "1 2 4 8"

bash examples/profiling/profile_attention_chunked_prefill.sh \
  --attention-backend VLLM_ROCM \
  --model Qwen3.8-2.4T-A95B-Quark-MXFP4 \
  --device mi355x \
  --profile-method device_event \
  --tp-sizes "1 2 4 8"

bash examples/profiling/profile_moe.sh \
  --model Qwen3.8-2.4T-A95B-Quark-MXFP4 \
  --device mi355x \
  --profile-method device_event \
  --tp-sizes "1 2 4 8" \
  --ep-sizes "1 2 4 8"
```

The attention wrapper above is a **prefill-only** campaign
(`--profile_only_prefill`). A complete simulator attention dataset also needs
decode measurements. Collect decode separately with the direct attention
producer and `--profile_only_decode --attention_backend VLLM_ROCM`, using a
separate output directory, then combine and validate phase coverage before
training. The wrapper's general-purpose `NO_OP` default remains useful for smoke
checks; the explicit native backend above is required for measured ROCm data.
Native acceptance checks compare actual attention output with a causal reference
and require a recorded attention timing sample; a DEVICE_EVENT label alone is
not evidence that an attention kernel ran.

For standard GDN phases, use the dedicated vLLM producer:

```bash
torchrun --standalone --nproc-per-node=8 \
  -m frontier.profiling.gdn.main \
  --model Qwen3.8-2.4T-A95B-Quark-MXFP4 \
  --model-path /path/to/local/checkpoint \
  --device mi355x \
  --profile-method device_event \
  --tensor-parallel-size 8 \
  --output-dir data/profiling
```

TP8 requires eight visible compatible devices and one process per rank; the
`torchrun` command launches all eight ranks. For a standalone TP1 campaign on
one visible device, use a separate output root:

```bash
python -m frontier.profiling.gdn.main \
  --model Qwen3.8-2.4T-A95B-Quark-MXFP4 \
  --model-path /path/to/local/checkpoint \
  --device mi355x \
  --profile-method device_event \
  --tensor-parallel-size 1 \
  --output-dir data/profiling-tp1
```

The TP8 output is
`data/profiling/compute/mi355x/Qwen3.8-2.4T-A95B-Quark-MXFP4/gdn.csv`.
Each row records the phase (`prefill` or `decode`), physical batch features,
GDN state/layout identity, runtime backend, rank aggregation, and component
timings. A one-token continuation remains a prefill input. Native profiling and
training still reject mixed prefill/decode rows. The simulator temporarily
predicts mixed-batch GDN work with prefill estimators and emits a warning,
particularly when **co-location** scheduling combines new prefill and running
decode. This preserves scheduler behavior but does not establish native mixed
execution or timing accuracy. See the [GDN prediction limitation](../training/README.md#standard-gdn).

## Validation and training handoff

Before training, inspect the output directory and validate its metadata:

```bash
bash examples/profiling/smoke_metadata.sh \
  --data_path data/profiling/compute/mi355x/Qwen3.8-2.4T-A95B-Quark-MXFP4
```

Train GDN artifacts with `measurement_type DEVICE_EVENT` and the same model,
device, TP, architecture profile, quantization signature, and runtime-stack
identity used by the producer. `GDNTrainer` writes six task estimators and a
manifest containing a SHA-256 fingerprint of `gdn.csv`. The simulator rejects
the cache if the configured CSV bytes later change.

Run the non-dummy simulator smoke only after the required CSVs are present:

```bash
bash examples/profiling/smoke_simulator_dense_csv.sh \
  --metrics-output-dir outputs/mi355x-dense
bash examples/profiling/smoke_simulator_moe_csv.sh \
  --metrics-output-dir outputs/mi355x-moe
```

These checks establish CSV loading and simulator control flow. They do not
replace measured GPU validation of the producer kernels.

## Separate experimental evidence

SGLang primitive replay and Kineto trace import remain under
`frontier/profiling/experimental/sglang/`. Replay records isolated
`HIP_GRAPH_REPLAY` artifacts; trace import writes `gdn-trace-summary.csv/json`.
Neither is a standard `DEVICE_EVENT` row, and neither is discovered by the
standard trainer. Routed replay uses Frontier's shared routing helper or an
explicit expert-count JSON. It does not observe a live router or capture a
whole model.

RCCL collection is also standalone:

```bash
python -m frontier.profiling.collectives.main \
  --disable_ray --num_gpus 8 --precision BF16 \
  --output_dir outputs/mi355x-rccl
```

Its output is collective evidence and is not a compute predictor CSV.

## Current evidence boundary

The current execution worktree has no AMD/MI355X worker. CPU checks cover
argument parsing, lazy imports, planning, metadata, schema validation, and
artifact identity. Actual HIP/ROCm event capture, vLLM GDN execution,
AITER/MXFP4 MoE execution, RCCL collectives, and SGLang graph replay remain
**SKIP: AMD/MI355X hardware unavailable**. Do not use synthetic BF16 values or
CPU schema success as benchmark or groundtruth parity.
