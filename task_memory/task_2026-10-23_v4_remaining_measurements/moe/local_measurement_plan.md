# V4 local-expert (MXFP4/MARLIN) measurement plan — 1×H800

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Created by subagent W128-V4-MOE-GRID. Read-only source/evidence inspection plus CPU `--dry_run`; no source, test, GPU, Docker, cloud or Git change. Root owns the final grid. |
| 2026-10-23 | Marked the interrupted three-point exploration historical; corrected V2/population evidence is maintained in `../moe_grid_design.md`. |

**Historical provisional exploration, not an executable run approval or accepted grid.** The delivering exploration ended in provider error30008. Its arithmetic, equal-rank claims, three-point grid and one-method plan below are superseded by `../moe_grid_design.md`: EP16 rank0 gathered populations16/32/6152/12288, separate CUDA_EVENT and record_function invocations, retained but unconsumed gate/shuffle columns, and no native timing/serving qualification. Root still owes a fresh package, image/SDK checks and execution evidence. The original dry-run evidence is retained, not used as final acceptance.

Task ID `W128-V4-MOE-GRID`. Scope: local measured expert-operator cost only; no dispatch,
communication, topology or admission claim.

## 1. Observed basis

- Native06 engine configuration (`native06_case_20261023/inputs/engine.json`): `tensor_parallel_size=8`,
  `data_parallel_size=2`, `enable_expert_parallel=true`, `max_num_batched_tokens=8192`,
  `max_num_seqs=128`, vLLM `0.28.0`, image `hub.i.basemind.com/vllm-openai/vllm-openai@sha256:2286e853…`,
  source `2cf0a69`. No `all2all_backend` is set, so the pinned default `allgather_reducescatter`
  applies (`vllm/config/parallel.py:188`); with TP8>1, DP2>1 and EP on, `use_sequence_parallel_moe`
  is true (`:673-682`), so the MoE input is SP-sliced to `ceil(T_lane/8)` rows per rank.
- `DPMetadata.sp_local_sizes` sets `local_sizes = repeat(ceil(tokens_lane/sp), sp)` per DP lane
  (`vllm/forward_context.py:60-69`, `:102-113`; applied at `runner/moe_runner.py:631`), so native06
  has 16 EP-ordered entries of 768 (prefill) or 1/2 (decode).
- Native06 serving populations (`driver/native06_window_basis.json`, all 32 rank files):
  prefill `NONE`, `num_tokens=6144`, one request per DP lane; decode `FULL`+`PIECEWISE` descriptors
  `num_tokens ∈ {1,2,4,8,16}` per DP lane, 1..11 scheduled requests per lane. Descriptors are the
  DP-synchronised padded key, and are not expert-GEMM row counts.
- Profiler routing (`frontier/profiling/moe/moe_wrapper.py:337-429`): with EP>1 it routes
  `num_tokens` rows over all 256 global experts and keeps rank 0's contiguous 16-expert map
  (`:294-296` always rank 0), so the local GEMM load is `6*num_tokens/256` rows per local expert.
- The timed kernel is `MarlinExperts.apply` with `global_num_experts=256` and `expert_map`
  (`moe_vllm_kernel.py:952-1073` → `mxfp4_marlin_experts.py:86-118`), i.e. exactly the
  all-gathered rows plus the expert map the native all-gather/reduce-scatter dispatch produces.

## 2. Population identity (inferred, arithmetic from §1)

One profiler `num_tokens` is one EP-group all-gathered population, `sum(local_sizes)`:
| Native window | lane tokens | rows/rank | profiler `num_tokens` | local rows (rank 0) | per local expert |
| --- | --- | --- | --- | --- | --- |
| prefill `NONE` 6144 | 6144 | 768 | 12288 | 4608 | 288 |
| decode descriptor ≤ 8 | 1/2/4/8 | 1 | 16 | 16 | 1 |
| decode descriptor 16 | 16 | 2 | 32 | 16 | 1 |

`num_tokens` is *not* the LANE token count (6144 would under-state per-expert rows 16×) and *not*
a padded descriptor; any lane population of 1..16 tokens maps to 16 or 32, so decode needs no row
per descriptor value.

## 3. Grid: three `num_tokens` at one topology

`EP=16` (local width 16, admitted P0), `TP=1` (checkpoint-MXFP4 shards by EP only;
`moe_vllm_kernel.py:869-875` rejects TP>1), `balanced` distribution, one sample. EP8 or extra token
points are excluded: the adapter always profiles rank 0 of EP16, so EP8 would measure a 32-expert
kernel that native P0 never runs.

| `num_tokens` | status | evidence |
| --- | --- | --- |
| 12288 | observed population | prefill 6144/lane × DP2, formula `16*ceil(6144/8)` |
| 32 | observed population | decode descriptor 16/lane → `16*ceil(16/8)`; largest align input (192 ids) |
| 16 | observed population | decode descriptors 1/2/4/8/lane → `16*ceil(d/8)` |
| (excluded) 22, 16384, 8192, EP8 | inferred-none | not present in native06; 22 wrongly assumes no SP sharding |

## 4. Commands (locked image, 1 H800, source-mounted)
Run in the native06 image with this worktree mounted and `PYTHONPATH` set to it; `--device h800`
only names the CSV directory and `-y` skips the interactive confirmation.

Smoke (single point, seconds):

```
python -m frontier.profiling.moe.main --models deepseek-ai/DeepSeek-V4-Flash \
  --device h800 --num_gpus 1 --disable_ray -y --num_tensor_parallel_workers 1 \
  --expert_parallel_sizes 16 --max_tokens 32 --num_tokens_list 32 \
  --load_distributions balanced --num_samples_per_distribution 1 \
  --routing_runtime_path balanced_topk --profile_method record_function \
  --output_dir /mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18/dsv4_moe_marlin_p0_20261023/smoke
```

Full grid (only after the §6 gates pass; fresh directory): same command with
`--max_tokens 12288 --num_tokens_list 12288 32 16` and `…/grid` as `--output_dir`.

CPU dry-run already executed with `frontier-py310` (no torch installed, torch never imported):
`/data/ycfeng/tmp/glm53_dsv4/w128_v4_moe_grid_20261023/dry_run_{smoke,grid}.json` report
`num_tokens [12288, 32, 16]`, `cases_per_parallel_point 1`, `profiled_steps 3`, EP16→16 local
experts; `grid_predict.py` → `grid_predict.json` predicts local counts 288×16 (prefill) and 1×16
for both decode points (torch-free, same feature code the profiler writes).

## 5. Output contract

One CSV, one row per `num_tokens` (3 rows), four operator groups per row:
`…/compute/h800/deepseek-ai/DeepSeek-V4-Flash/moe_kernel_only.csv`
(`build_profile_method_output_path`, `frontier/profiling/utils/__init__.py:126-147`;
written at `frontier/profiling/moe/main.py:1137-1166`).

- `time_stats.{moe_gating_linear,moe_gating_routing_topk,moe_shuffling,moe_grouped_gemm}.{min,max,mean,median,std,count,kernel_count}`.
- Identity/features: `num_tokens`, `num_experts`, `num_experts_per_device`,
  `expert_parallel_size`, `router_topk`, `hidden_dim` (4096), `expert_hidden_dim` (2048),
  `use_gated`, `num_tensor_parallel_workers`, `routing_{runtime_path,assignment_policy,
  weight_policy,uses_router_logits}`, `gating_runtime_context{,_impl}`, `total_routed_tokens`,
  `model_expansion_ratio`, `tokens_per_expert_avg`, `tokens_to_experts_ratio`,
  `expert_utilization`, `min_load_ratio`, `load_imbalance_cv`, `max_load_ratio`, `load_entropy`,
  `load_gini_coefficient`, `load_distribution`, `seed`, `moe_grouped_gemm_backend`
  (`vllm_marlin_mxfp4`), `moe_quantization_mode` (`mxfp4`), `moe_native_backend` (`MARLIN`),
  `typed_operator_contracts`, `measurement_type` (`KERNEL_ONLY`), `profiling_precision`,
  `model_arch`, `model_architecture_profile`, `quant_signature`.
- No `time_stats.moe_grouped_gemm.num_tokens_post_padded`/`block_size_m`/`block_size_m_ranges`:
  the checkpoint-MARLIN branch never calls `try_get_optimal_moe_config`
  (`moe_vllm_kernel.py:952-1073`), so V4's consumer needs the load-feature row set this emits
  (`moe_dataset_training.py:53-79`, `deepseek_v4.py:790-822`).
- `record_function`=KERNEL_ONLY only; a CUDA_EVENT run is a second invocation writing a different
  file (`…/moe.csv`) and is not part of this grid.

## 6. Acceptance gates (smoke first)
1. stdout contains `vLLM 0.28.0 loaded successfully (API: functional_fused_experts)`.
2. No `RuntimeError` from `select_deepseek_v4_mxfp4_moe_backend` (`mxfp4_marlin_experts.py:188-192`):
   the oracle must still pick `MARLIN`, not `BATCHED_MARLIN` (deepep/nixl only,
   `fused_moe/config.py:1087-1088`; a non-default `VLLM_ALL2ALL_BACKEND` would flip it).
3. CSV has 1 row, `moe_native_backend=MARLIN`, non-zero `median`/`kernel_count` for all four
   operators, `num_tokens=32`, `total_routed_tokens=16`, `tokens_per_expert_avg=1.0`,
   `load_entropy≈4.000000000017834` (exact float; consumers load with `round_trip`).
4. Only then run §4 full grid and check 3 rows with the §2 populations (4608/16/16 local rows).

## 7. Symbols to re-verify inside the locked image

`vllm/model_executor/layers/fused_moe/`: `fused_experts`, `utils._resize_cache`,
`config.{FusedMoEConfig,FusedMoEParallelConfig,RoutingMethodType.Unspecified}`,
`activation.MoEActivation.SILU`, `oracle/mxfp4.{select_deepseek_v4_mxfp4_moe_backend,
convert_weight_to_mxfp4_moe_kernel_format,make_mxfp4_moe_quant_config}`,
`experts/marlin_moe.MarlinExperts.{moe_problem_size,workspace_shapes,workspace_dtype,apply}`;
plus `vllm.config.{VllmConfig,get_current_vllm_config,get_current_vllm_config_or_none,set_current_vllm_config}`,
`vllm.distributed.{init_distributed_environment,ensure_model_parallel_initialized}`,
`vllm.utils.network_utils.get_open_port` — all present in source `2cf0a69`; the installed wheel
hash is the remaining unknown.

## 8. Shape limits stated explicitly
- Gating vs GEMM: the row's `num_tokens` serves the expert GEMM (`16*ceil(T_lane/8)`) but feeds
  `moe_gating_linear`/`moe_gating_routing_topk` the same 12288/32/16 rows, while the native router
  runs on `ceil(T_lane/8)` = 768/1/2 rows per rank. Those gating columns over-state the native
  rank-local router 16× and must not be consumed as native P0 gating cost (V4's structural path
  charges gating as a dummy baseline, `deepseek_v4.py:296`). A native-shaped gating row needs a
  separate case at 768/1/2, which would mis-state the expert GEMM; root decides if any consumer
  needs it.
- Rank/load: only EP rank 0 (experts 0..15) is measured; under balanced routing every rank has the
  same per-expert load (equal modulo remainder) and the kernel is expert-agnostic, so rank choice
  changes ids, not shapes. 12288 exceeds `max_num_batched_tokens=8192`: it is an all-gathered
  population, not a schedulable batch.

## 9. Remaining runtime unknowns
Per-point wall time and HBM (expected ≤2 GiB of packed FP4 weights, seconds per point) are
unmeasured; whether the marlin apply at 16 rows keeps a constant kernel count across the 20
`record_function` steps (the tracer raises on variation) is untested; whether the image's installed
vLLM exposes the §7 symbols exactly (random packed FP4 weights must also select the same marlin
template as checkpoint weights), and whether `swiglu_limit=10.0` clamp parity holds in-kernel.
