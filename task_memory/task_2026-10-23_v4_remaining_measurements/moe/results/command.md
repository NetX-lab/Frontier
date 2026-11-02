# W128-MOE01-DATA commands

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded CPU-only audit commands and unchanged producer command provenance. |

## Local audit (executed)

Working directory: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. The system scope applies `MemoryMax=30G`; `runuser` runs the CPU-only adapter as `brainpp`. No Torch/vLLM import, GPU, network, platform, cloud, Docker, or Git operation occurs.

```bash
sudo -n systemd-run --scope -p MemoryMax=30G /usr/sbin/runuser -u brainpp -- /usr/bin/env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 timeout 60s /usr/bin/python3 /data/ycfeng/tmp/glm53_dsv4/W128-MOE01-DATA/audit.py /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-23_v4_remaining_measurements/moe/results/data_audit.json
```

Final adapter: 209 lines, SHA256 `9f15d8dc4d2397ade5d6e3efa181100cee76ea1ae1e03f6a252ca49e5fdfaf28`. It executes unchanged AST-selected `RecordFunctionTracer` CPU parsing methods, `MoELoadImbalanceInput`, `num_tokens_post_padded`, `RuntimeIdentity`, and the native Marlin block-selection loop. It does not execute profiler entry/exit or materialize device tensors.

```bash
sudo -n systemd-run --scope -p MemoryMax=30G /usr/sbin/runuser -u brainpp -- /usr/bin/env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 timeout 60s /usr/bin/python3 /data/ycfeng/tmp/glm53_dsv4/W128-MOE01-DATA/verify.py
```

The final integrity receipt is `/data/ycfeng/tmp/glm53_dsv4/W128-MOE01-DATA/output_hashes.json`. It verifies all hash-bound inputs and records SHA256/size of all three delivered files.

## Retained producer commands (provenance only; not executed locally)

`h800_moe_local_01/jobs/moe.json` contains the complete two argv arrays, including source/model revisions and distinct output directories. The measured worker used Python 3.12.3. The common command is:

```bash
python3 -m frontier.profiling.moe.main --models deepseek-ai/DeepSeek-V4-Flash --device h800 --num_gpus 1 --disable_ray --num_tensor_parallel_workers 1 --expert_parallel_sizes 16 --num_tokens_list 12288 6152 32 16 --max_tokens 16384 --load_distributions balanced --num_samples_per_distribution 1 --routing_runtime_path balanced_topk --yes --source_revision 56e9405dad76d30d4ada9d5bdab124f898bf5065 --model_config_revision 0d992cbeef1f2d4bd92a8ae67a97a43df0c1bce5 --profile_method cuda_event --output_dir /tmp/glm_producers/out/EP16_RANK0_LOCAL_MOE_CUDA_EVENT
python3 -m frontier.profiling.moe.main --models deepseek-ai/DeepSeek-V4-Flash --device h800 --num_gpus 1 --disable_ray --num_tensor_parallel_workers 1 --expert_parallel_sizes 16 --num_tokens_list 12288 6152 32 16 --max_tokens 16384 --load_distributions balanced --num_samples_per_distribution 1 --routing_runtime_path balanced_topk --yes --source_revision 56e9405dad76d30d4ada9d5bdab124f898bf5065 --model_config_revision 0d992cbeef1f2d4bd92a8ae67a97a43df0c1bce5 --profile_method record_function --output_dir /tmp/glm_producers/out/EP16_RANK0_LOCAL_MOE_RECORD_FUNCTION
```

The producer's worker-local `/tmp/glm_producers` paths above are archived provenance, not local audit write targets.
