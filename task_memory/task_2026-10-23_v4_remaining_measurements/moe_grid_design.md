# W128 — native V4 routed-MoE row population and the one-GPU producer grid

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Initial corrected design. Supersedes the V1-dispatcher draft of the W128 report (wrong runner generation, wrong gate-row factor, over-strong equivalence claims). |
| 2026-10-23 | Followup: gate/shuffle consumption wording replaced with retain-data/no-consume. The accidental worktree-root copy is historical; this task path is maintained. |
| 2026-10-23 | Root corrected asymmetric gating factors, SP-versus-graph padding, rank-specific load and premature qualification language; limited the table to the four selected shapes. |

Native source `/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69` (`2cf0a69`), producer `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/profiling/moe/`. Marks: **[S]** source fact, **[R]** needs runtime, **UNKNOWN** not established.

## 1. Native chain (V2 runner)

| Step | Evidence |
| --- | --- |
| Runner is V2: `vllm.v1.worker.gpu.model_runner.GPUModelRunner` | retained JSON `inventory.model_runner` |
| Batch descriptor synced across DP: AR of `num_tokens` per rank, then `synced_num_tokens = max(...)`, re-`dispatch`, `num_tokens_across_dp[:] = synced_desc.num_tokens` | `vllm/v1/worker/gpu/dp_utils.py:19-95` (`:76-91`), called at `model_runner.py:1456` |
| Descriptor `num_tokens` is a **graph** size: `_candidates` maps every `i` in `(prev_capture, capture]` to the descriptor of the next capture size | `vllm/v1/worker/gpu/cudagraph_utils.py:173-300` (`:283-295`), `dispatch` `:382-404` |
| If the synced cg mode is NONE (any rank eager, or no captured key) there is **no token padding**: the rank keeps its own `num_tokens` and the raw per-rank vector | `dp_utils.py:69-71` |
| `num_tokens_after_padding = batch_desc.num_tokens`; `positions = positions[:num_tokens_after_padding]` | `model_runner.py:1109`, `:1287-1289` |
| `full_num_tokens = positions.shape[0]`; `sp_shard` pads to TP=8 then takes `ceil(N/8)` contiguous rows per TP rank | `models/deepseek_v4/nvidia/model.py:1149-1157`; `models/common/ops/sequence_parallel.py:42-50` |
| Gate runs on the **local L rows, before any EP communication**; modular path selects experts, then `forward_modular` | `fused_moe/runner/moe_runner.py:862-878`, `:586-600` |
| Router is the simulator when `VLLM_MOE_ROUTING_SIMULATION_STRATEGY` is set (retained `inventory.moe_routing.environment = "balanced_topk"`) | `router/router_factory.py:114-120`; `routing_simulator_router.py:296-321` |
| `balanced_topk`: `ids = (6*row + j + 6*token_offset) % 256`, `token_offset = sum(local_sizes[:ep_rank])`, routes **every physical row incl. SP/graph padding** (no `-1` filter) | `frontier/profiling/moe/balanced_routing.py:11-35`, `:52-70` |
| Dispatch = `all_gatherv` over the 16-rank EP group with `sizes = dp_metadata.local_sizes` (retained `all2all_backend = allgather_reducescatter`) | `distributed/device_communicators/all2all.py:45-70`, `:95-141`; `prepare_finalize/naive_dp_ep.py:158-164` |
| `local_sizes = repeat_interleave(ceil(N_d/8), 8)` over DP ranks | `forward_context.py:61-69`, `:101-113` |
| Marlin: `block_size_m` from `M_est*6/16/bm < 0.9`, `moe_align_block_size(..., 256, expert_map, ignore_invalid_experts=True)`, 2 GEMMs, `moe_sum` | `experts/marlin_moe.py:328-341`, `:135`, `:202`, `:393` |
| Zero-count expert contributes0 logical padded expert tiles (`ceil(count/bm)*bm`); this does not establish a zero launch grid or zero rank latency for alignment/GEMM/sum | `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu:163-171`; physical launches/time remain runtime evidence |
| Expert map: contiguous 16 experts per EP rank, `-1` elsewhere | `expert_map_manager.py:60-92` |

So the token transforms are **[S]**: `N_d` (per DP rank, graph-padded) → `L_d = ceil(N_d/8)` → gate rows `L_d` on each of that rank's 8 TP shards → `M = 8 * Σ_d L_d` rows at the Marlin kernel on **all 16 ranks**.

## 2. Corrected arithmetic

Equal shards give `M = 16*L` and therefore **gating rows per rank = L = M/16** (the earlier `M/8` was arithmetically wrong).

| native step (retained) | per-rank `N_d` | `L_d` | `M` | gate rows/rank | expert pairs | pairs/expert | `block_size_m` | align rows **per active rank** | inactive ranks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode 1/2/4/8 | descriptor 1/2/4/8; each SP-padded to 8 | 1,1 | 16 | 1 | 96 | 0..1 | 8 | 128 (6 active) | 10 ranks, 0 expert assignments |
| decode 16 | 16 | 2,2 | 32 | 2 | 192 | 0..1 | 8 | 128 (12 active) | 4 ranks, 0 expert assignments |
| prefill 6144+1 | 6144 / 1 | 768 / 1 | 6152 | 768 / 1 | 36912 | 144/145 | 64 | 3072 (all 16) | — |
| prefill 6144+6144 | 6144 | 768,768 | 12288 | 768 | 73728 | 288 | 64 | 5120 (all 16) | — |

Kernel-time shape is therefore `(16 expert blocks × ceil(count/bm) blocks of bm rows)`: `bm=8` below `M_est≈19` (`M≲304`), 16 at 512, 64 from 6152. Whether that makes per-rank time flat across `M=16..304` is a **hypothesis [R]**, not a source consequence — the grid shape is constant but occupancy, EP-rank activity and grid-launch cost are not.

## 3. Evidence (retained)

Window basis: `task_memory/task_2026-10-23_v4_remaining_measurements/driver/native06_window_basis.json`.
Per-rank: `/data/ycfeng/tmp/glm53_dsv4/native06_case_20261023/runs/groundtruth_clean/dsv4-h800-p0-canonical-20261021-06/nodes/<host>/serving/rank<N>.json`, keys `inventory.{model_runner,effective_graph,resolved.parallel_config.all2all_backend,moe_routing.environment}`, `inventory.initialization.backends["model.layers.0.ffn.experts.routed_experts"] = {backend: MARLIN, experts: MarlinExperts, input_dtype: None, prepare_finalize: MoEPrepareAndFinalizeNaiveDPEPModular}`, and `counters.rows[].{batch_descriptor, dispatch_evidence.dispatch.{descriptor,num_tokens_across_dp}, input_shapes.input_ids, kind, phase}`.

- `M=16`: 240 `input_shapes.input_ids=[1]` + 384 `[2]` + 816 `[4]` + 1584 `[8]` rank-rows, all with `num_tokens_across_dp == (N,N)` and descriptors `num_tokens ∈ {1,2,4,8}` (`PIECEWISE` and `FULL`/uniform variants).
- `M=32`: 1040 `[16]` rank-rows with `(16,16)`, `FULL`/`PIECEWISE` `num_tokens=16`.
- `M=12288`: 16 rank-rows `[6144]` with `(6144,6144)`, `cg_mode=NONE`.
- `M=6152`: 96 rank-rows `[6144]` with `(6144,1)` or `(1,6144)`, `cg_mode=NONE`; plus 16 `kind="dummy"` rank-rows with `[1]` whose vector carries the peer's `6144`. The light side of that mixed step is a **dummy** forward, so `M=6152` is an executed shape whose provenance is not steady-state serving.

The DP vector is the **cross-DP joined** AR result replicated on both nodes; `input_shapes` is the per-rank observation. No retained row joins two different local populations, so the mixed step is evidenced by the vector plus the dummy rows, not by one shared ordinal.

## 4. Producer semantics (what `--num_tokens` means)

- For `EP=16` the producer generates ids over the **global 256** experts on `num_tokens` rows and maps rank 0's contiguous block (`moe_wrapper.py:352-364`, `:294-320`), so `--num_tokens` at `profile_grouped_gemm` is **M**, the row count at the expert kernel.
- `profile()` passes the same `M` to gating, shuffling and grouped GEMM (`moe_wrapper.py:872-891`), while native gating receives `L_d`. Equal shards give `L=M/16`; the asymmetric6152 case instead has `L_d=768` or1. One producer row therefore cannot price both native gating and gathered expert compute. Retain all original columns; do not consume gating columns for these native tuples.
- `--load_distributions balanced` yields `ids[r,k] = (6r+k) % 256` (`load_distribution.py:86-89`, `moe_impl.py:82-103`). Native prefix offsets `sum(local_sizes[:rank])` reconstruct the contiguous global row index after EP-order gathering. This supports routing-index algebra over the gathered `M`, not input/timing equivalence. Individual expert counts are floor/ceil(`6M/256`); rank loads are their local16-expert sums and can differ, including zero at decode. Offsets do not change the static expert ownership; rank0 data do not qualify every EP rank.
- That algebra is **not** labelled by `load_distributions`: the row's only routing identity is the gating metadata column. `--routing_runtime_path` (choices incl. `balanced_topk`, `main.py:320-334`; profile value is `None` for `deepseek_v4` so the CLI applies, `moe_wrapper.py:124-129`) selects the gating path (`moe_impl.py:226-229`) and stamps `routing_assignment_policy=global_round_robin_balanced`, `routing_uses_router_logits=False` (`moe_impl.py:268-281`). Without it a `balanced`-labelled grouped-GEMM row is indistinguishable from a `uniform_topk` shuffle, and the gating row is a real `fused_topk` over random logits.
- Backend lineage: same selector `select_deepseek_v4_mxfp4_moe_backend`, asserted `MARLIN`, same `MarlinExperts` class, EP width 16, `global_num_experts=256`, `expert_map`, `apply_router_weight_on_input=False` (`mxfp4_marlin_experts.py:186-193`, `:86-111`). Weight byte counts match the retained `[16,256,8192] int32` / `[16,128,8192]` (134,217,728 B and 67,108,864 B), but **byte count does not establish content or timing parity**, and the weights are random packed bytes. Not claimed: numerical or timing equality.

## 5. Timed scope (avoid double counting)

`moe_grouped_gemm` (`mxfp4_marlin_experts.py:86-111`) covers, inside the timer: `moe_align_block_size` (sort + per-expert block padding), both Marlin GEMMs, and `moe_sum` with `expert_map`. Outside the producer timer: workspace/`empty_like` allocation (`moe_vllm_kernel.py:966-972`) — natively that allocation is inside the step (`modular_kernel.py:1113-1160`). Not covered anywhere in this row: AG dispatch, RS combine, gate, shared experts, output truncation.
`moe_shuffling` re-runs `moe_align_block_size` standalone with its own block-size heuristic (`moe_impl.py:369-386`, `:388-398`), so it overlaps `moe_grouped_gemm`; never sum the two. The candidate measurement scope is the existing rank0 `moe_grouped_gemm` envelope. Retain every original column; gating and standalone shuffle columns are not consumed for this scope. GPU evidence and independent data acceptance remain necessary before any cost qualification.

## 6. Minimal grid (EP16 only)

```
python -m frontier.profiling.moe.main --models deepseek-ai/DeepSeek-V4-Flash \
  --device h800 --num_gpus 1 --disable_ray --num_tensor_parallel_workers 1 \
  --expert_parallel_sizes 16 \
  --num_tokens_list 12288 6152 32 16 --max_tokens 16384 \
  --load_distributions balanced --num_samples_per_distribution 1 \
  --routing_runtime_path balanced_topk --profile_method cuda_event --yes
```
`--num_tokens_list` bypasses `--max_tokens`'s grid but is still validated against it (`utils/__init__.py:189-212`), hence `--max_tokens 16384`. `--profile_method` accepts `cuda_event` (2 warmup + 20 event samples, `moe_vllm_kernel.py:682-705`) or `record_function` (20 iterations under `record_function("vidur_moe_grouped_gemm")`, `:708-731`; needs `--output_dir`); `cuda`/`kernel_only` normalize to these (`utils/__init__.py:44-58`). TP must be 1 (`moe_vllm_kernel.py:871-876`).

Rows: `12288` = symmetric prefill, `bm=64`, 5120 rows/rank; `6152` = mixed prefill+dummy (conditional provenance); `16` and `32` = the two observed decode populations (`bm=8`, 128 padded rows on6 and12 active ranks respectively). No extra48/512/6144 points are selected. Capture/profile capacity does not establish serving populations.

## 7. Open / not proven

**[R]** per-rank time behaviour vs `M`; cost of the dummy light side; whether live decode ever exceeds 16 tokens (nothing retained does); whether the real-router histogram matters for calibration. **[UNKNOWN]** timing equivalence of random packed weights; whether the 43-layer stream/L2 interaction makes single-layer rows additive — no per-layer × 43 claim is made.
