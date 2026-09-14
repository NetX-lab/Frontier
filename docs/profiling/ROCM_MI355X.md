# ROCm and MI355X Profiling

## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
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
├── linear_op.csv
├── attention.csv
├── moe.csv
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

For standard GDN phases, use the dedicated vLLM producer:

```bash
python -m frontier.profiling.gdn.main \
  --model Qwen3.8-2.4T-A95B-Quark-MXFP4 \
  --model-path /path/to/local/checkpoint \
  --device mi355x \
  --profile-method device_event \
  --tensor-parallel-size 8 \
  --output-dir data/profiling
```

The output is
`data/profiling/compute/mi355x/Qwen3.8-2.4T-A95B-Quark-MXFP4/gdn.csv`.
Each row records the phase (`prefill` or `decode`), physical batch features,
GDN state/layout identity, runtime backend, rank aggregation, and component
timings. A one-token continuation remains a prefill input. Same-batch prefill
and decode is rejected because the standard trainer requires one
phase-qualified estimator.

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
