# G5h: H200 collective curves for DeepSeek-V4 (design §5 stage 2)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Corrected the dispatch `topk_ids` dtype to int64 (§2, §3, §4), found by the G5h-16 kernel comparison ([../../test_report_2026-10-28_b10_g5h16.md](../../test_report_2026-10-28_b10_g5h16.md)). |
| 2026-10-28 | Created: stage order, the two native-path findings, the V4 forward call list, the grid, the launch and runtime checks, and the split into Codex assignments B9 and B10. |

## 1. Purpose and order

Design §5 runs stage 2 (producers and grids) in parallel with stage 3 (simulator files). H200 has no collective curves, so G5h is on the critical path of the H200 milestone ([../design.md](../design.md) §5). G5h therefore goes first among the stage 2 jobs. The simulator entries 6-12 continue on CPU while G5h runs.

Order: `B9 (V4 call list, native SP ops, grid, launcher; CPU) -> review and commit -> G5h-16 (16 H200) -> {B10 (native kernel table and comparator; CPU), lead review of G5h-16 data} -> G5h-32 (32 H200, soft gates)`. B10 runs while G5h-16 runs and must finish before G5h-32 is launched, so G5h-16's kernels are compared with the native traces before 32 GPUs are spent.

## 2. Evidence: how native vLLM issues the V4 collectives

Source: native commit `2cf0a69` checkout `/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm`. Runtime: image `hub.i.basemind.com/vllm-openai/vllm-openai@sha256:2286e853…`, NCCL 2.30.7 (P2 server log), the native P2 engine (TP8, DP2 or DP4, EP on, PP1, bf16, async scheduling, default all2all backend `allgather_reducescatter`).

**Finding 1: the V4 sequence-parallel collectives take vLLM's custom kernels first.**
- `models/deepseek_v4/nvidia/model.py:962-967` calls `sp_all_gather(x)[: positions.shape[0]]` before attention and `sp_reduce_scatter(x)` after it, in every layer. `model.py:1196-1197` calls `sp_all_gather(hidden_states)[:full_num_tokens]` once after the last layer, on the mHC tensor `[rows, hc_mult, hidden]`.
- `models/common/ops/sequence_parallel.py`: `sp_all_gather` first tries `get_tp_group().device_communicator.custom_all_gather(x)` and falls back to `tensor_model_parallel_all_gather(x, 0)`. `sp_reduce_scatter` first pads the rows to a multiple of TP with `F.pad`, then tries `custom_reduce_scatter`, then falls back to `tensor_model_parallel_reduce_scatter(x, 0)`.
- `custom_all_reduce.py:399-500`: the custom path applies when the input is bf16/fp16/fp32, contiguous, a multiple of 16 bytes, and within the size limit (MNNVL all-gather 2 MiB for 8 ranks, MNNVL reduce-scatter 16 MiB). With the MNNVL multicast buffer it runs `mnnvl_lamport_all_gather` / `mnnvl_lamport_reduce_scatter`; larger inputs fall back to NCCL.
- The P2 server log confirms "Using the MNNVL Lamport all-gather kernel" and "… reduce-scatter kernel", and TP all-reduce backends `['CUSTOM', 'SYMM_MEM', 'PYNCCL']`; the census shows `mnnvl_lamport_all_gather` 44 and `mnnvl_lamport_reduce_scatter_kernel` 43 per dummy-prefill forward ([../evidence/runtime_census/p2_32.txt](../evidence/runtime_census/p2_32.txt)).
- Consequence: the existing benchmark's `all_gather` / `reduce_scatter` ops call `GroupCoordinator.all_gather` / `reduce_scatter`, which never take the custom path. They would measure NCCL kernels where native runs the Lamport kernels. B9 adds op kinds that call the native `sp_all_gather` / `sp_reduce_scatter` functions.

**Finding 2: the V4 EP dispatch is one grouped `all_gatherv`.**
- The P2 server log shows `MoEPrepareAndFinalizeNaiveDPEPModular` with `MarlinExperts` (MXFP4 weights, bf16 activations). Its `prepare` (`prepare_finalize/naive_dp_ep.py:113-170`) does not quantize the input (no input quantization dtype), so no scale tensor is gathered, and calls `get_ep_group().dispatch(hidden, topk_weights, topk_ids, is_sequence_parallel=True)`.
- `AgRsAll2AllManager.dispatch` (`all2all.py:95-130`) gathers `[hidden bf16, topk_weights f32, topk_ids]` in one `all_gatherv` over the EP group, with per-EP-rank sizes from `sp_local_sizes` (`forward_context.py:102-113`: each lane's `ceil(T / TP)` repeated TP times). `topk_ids` are int64: the measured runs route with the pinned balanced routing (`frontier/profiling/moe/balanced_routing.py:27` returns `torch.long` when `indices_type` is None), and the naive DP/EP prepare step leaves `indices_type` None (`prepare_finalize/naive_dp_ep.py:103-104`). The P2 census shows `moe_align_block_size_kernel<long>`. The int32 of the model router (`model.py:582`) does not apply, because the balanced routing replaces it. `combine` is one `reduce_scatterv` over the EP group with the same sizes.
- The census agrees: 129 `AllGatherV` kernels per dummy-prefill forward, 3 per layer (one per gathered tensor).
- Frontier prices three separate `dsv4_ep_*_allgather` operators today. Entry 8 replaces them with one grouped dispatch call priced from this measurement.

**Reduce-scatter key (decision).** The benchmark issues `sp_reduce_scatter` on the actual `T` rows, so the pad copy is measured as native runs it. The measurement is keyed by the unpadded input bytes `T × hidden × 2`; the padded bytes are `output bytes × TP`. Entry 8 changes `_dsv4_sp_attention_reduce_scatter_bytes` (today padded: `local_rows × hidden × 2 × tp`) to the unpadded key. For `T = 6144` both are equal.

**No DP sync call.** D2 (c) prices V4 host time as an explicit zero, and the consumer skips the DP-sync rows, so the V4 list has no DP-sync call.

## 3. The V4 forward call list

`T` = tokens of a lane, `r = ceil(T / 8)` SP rows per rank, hidden 4,096 bf16 (8,192 bytes per row), 43 layers, all MoE, `k = 6`.

| Call | Op | Group | Tensors per rank | Calls per forward | Source |
| --- | --- | --- | --- | --- | --- |
| `embedding_all_reduce` | `all_reduce` | TP | `[T, 4096]` bf16 | 1 | `VocabParallelEmbedding` |
| `sp_attention_all_gather` | `sp_all_gather` | TP | `[r, 4096]` bf16 | 43 | `model.py:962-963` |
| `sp_attention_reduce_scatter` | `sp_reduce_scatter` | TP | `[T, 4096]` bf16 before the pad | 43 | `model.py:966-967` |
| `sp_final_mhc_all_gather` | `sp_all_gather` | TP | `[r, 4 × 4096]` bf16 | 1 | `model.py:1196-1197` |
| `moe_dispatch_all_gatherv` | `all_gatherv` | EP | `[r, 4096]` bf16, `[r, 6]` f32, `[r, 6]` int64 | 43 | `all2all.py` dispatch |
| `moe_combine_reduce_scatterv` | `reduce_scatterv` | EP | input `[Σ r, 4096]` bf16 | 43 | `all2all.py` combine |
| `logits_all_gather` | `all_gather` dim −1 | TP | `[b, 16160]` bf16 | 1, eager | `logits_processor.py:126` |

The final gather's native shape is `[r, 4, 4096]`. The custom and NCCL paths depend only on the bytes, the dtype and contiguity, so the 2-D `[r, 16384]` tensor issues the same kernel. The logits columns are the padded vocabulary 129,280 divided by TP 8; the V4 config sets no `head_dtype`, so logits use the model dtype (`config/model.py:2304-2323`).

## 4. Grid

The points are those of [../c1_coverage.md](../c1_coverage.md) §4.5 with its margins.

- **Graph steps:** `T ∈ {1, 2, 4, 8, 16, 24, 32}`, equal on every lane (DP padding), captured. `T = 24, 32` are the L = 3, 4 margins.
- **Eager steps:** lane tokens from `{6144, 1}`. The TP-group calls run only where every lane has the same `T`: their cost depends on the own lane only, and each lane is on its own node. The EP calls run on every ordered lane vector (4 for DP2, 16 for DP4).
- **Logits:** eager, `b = 1-24` rows on every lane.
- **G5h-16** (TP8, DP2, EP16): every call above.
- **G5h-32** (TP8, DP4, EP32): only the two EP calls. The TP calls are the same TP8 groups on a node as in G5h-16.

Expected per-lane input bytes (C1 §4.5 plus the margins):

| Call | Graph | Eager |
| --- | --- | --- |
| embedding all-reduce | 8,192, 16,384, 32,768, 65,536, 131,072, 196,608, 262,144 | 8,192, 50,331,648 |
| SP attention all-gather | 8,192 (T 1-8), 16,384, 24,576, 32,768 | 8,192, 6,291,456 |
| SP attention reduce-scatter, unpadded key | 8,192, 16,384, 32,768, 65,536, 131,072, 196,608, 262,144 (padded 65,536 × r) | 8,192 (padded 65,536), 50,331,648 |
| SP final mHC all-gather | 32,768, 65,536, 98,304, 131,072 | 32,768, 25,165,824 |
| logits all-gather | — | 32,320 × b, b = 1-24 |
| EP dispatch, per lane: activation / weights / ids | r × 8,192 / r × 24 / r × 48, r = 1-4 | 6,291,456 or 8,192 / 18,432 or 24 / 36,864 or 48 |
| EP combine input (EP16) | 131,072 × r | 100,663,296; 50,397,184 (mixed) |
| EP combine input (EP32) | 262,144 × r | for example 151,060,480 for one 1-token lane |

Case counts: G5h-16 has 7 × 6 graph cases, 2 × 6 uniform eager cases, 2 × 2 mixed EP cases and 24 logits cases (82). G5h-32 has 7 × 2 graph cases and 16 × 2 eager EP cases (46).

## 5. Launch and runtime checks

- H200 `step_main`, non-preemptible, through the StepMind Python `RJobBackend` via `/ds4-exec` (Pi). One replica per node; `DISTRIBUTED_JOB=true`, `task_anti_affinities=["hard/default"]`, and the default RDMA `custom_resources` of `ResourceConfig` (not overridden). The pinned image and CUDA compatibility path of the native P2 runs; the same `VLLM_*` settings as the native runs (only the CUDA compatibility pair); `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,NET`; platform `NCCL_*` values unchanged.
- The per-node entry reuses the K5 rendezvous helpers (`w5_glm_comm_20261020/launch/node_entry.py`) without the UCX device selection, which only the NIXL stages need.
- Before `torchrun`, each node writes `runtime_check.json`:
  - sha256 of the image's `libnccl.so.2` against the P2 `overlay.json` value;
  - sha256 of the mounted benchmark sources and the stage config against the values the launcher recorded at submission, with the Git HEAD;
  - sha256 of the image's `vllm/models/common/ops/sequence_parallel.py`, `distributed/device_communicators/{cuda_communicator,custom_all_reduce,all2all}.py` and `distributed/parallel_state.py` against the `2cf0a69` checkout.
- Gate mode: in G5h-16 the first two checks are hard (the node fails before measuring); in G5h-32 every check is soft (recorded, the run continues; the 32-GPU soft-gate rule). The vLLM file comparison is always soft: the measurement is valid for the image either way, because the native runs used the same image; a difference means the lead re-reads the image's source before accepting the data.
- Outputs: `/mnt/codesign-exp/ycfeng/glm53_dsv4/comm/h200/<stage>_01/`.
- GPU-hours: estimate 0.5 h per job (C1 §5), G5h-16 about 8 and G5h-32 about 16. Worker timeouts 2,700 s each bound the worst case at 12 + 24 = 36 GPU-hours, the G5h cap.

## 6. Acceptance

- B9 (CPU): the planned V4 points equal §4's table; the GLM K5 dry runs stay byte-identical; unit tests with fakes pass; the existing collective tests pass.
- G5h-16: terminal `Succeeded`; `runtime_check.json` PASS on both nodes; every planned case measured in 3 passes; `queue_held_fraction` 1.0 or explained; the kernels per case recorded.
- B10 (CPU, before G5h-32): per (role, mode, lane token vector), the native per-forward collective kernel multiset from the P2 traces equals the multiset that the benchmark's `kernels.jsonl` gives for the same forward, and the kernel durations agree within the native spread. A difference is reported to YC before G5h-32.
- G5h-32: as G5h-16 with soft checks.
