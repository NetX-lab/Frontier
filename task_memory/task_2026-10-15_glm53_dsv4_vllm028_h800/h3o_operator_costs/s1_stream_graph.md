# S1: stream graph of the DeepSeek V4 attention layer and the MoE shared expert (H3o-(b) stage 0)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-27 | Created: source reading of the pinned vLLM V4 model, the stream graph per layer variant and runtime mode, the trace checks on the retained H200 P2-32, H200 P2-64 and H800 P0 traces, the MoE shared-expert finding, and the inputs entry 2 takes from S1. |

## 1. Scope and sources

S1 is the CPU-only step of [design.md](design.md) §5 stage 0. It gives the stream graph of each V4 attention variant from the vLLM source: which work runs on which stream, the event waits, and the branches that depend on the row count, the capture kind and the context length. Entry 2 (`dsv4_composition.py`) and the replay route of stage 1a use it.

- vLLM source: `V=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`, the pinned native engine of all three cases. Paths below are relative to `$V/vllm/`.
- Model: 43 layers. 2 uncompressed layers (compress ratio 1), 21 CSA layers (compress ratio 4, with an indexer) and 20 HCA layers (compress ratio 128, compressor without an indexer). Runner: V2 `GPUModelRunner`, graph mode `FULL_AND_PIECEWISE`.
- Traces: the retained native torch traces of TP rank 0 of DP lane 0.

| Case | Role | Trace |
| --- | --- | --- |
| H200 P2-32 | PREFILL | `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-profile-20261025-01/artifacts/torch_traces/prefill/dp0/dp0_pp0_tp0_dcp0_ep0_rank0.1792942682770529608.pt.trace.json.gz` |
| H200 P2-32 | DECODE | `.../dsv4-h200-p2-profile-20261025-01/artifacts/torch_traces/decode/dp0/dp0_pp0_tp0_dcp0_ep0_rank0.1792942701901162313.pt.trace.json.gz` |
| H200 P2-64 | DECODE | `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-64-profile-20261025-01/artifacts/torch_traces/decode/dp0/dp0_pp0_tp0_dcp0_ep0_rank0.1792946239761662380.pt.trace.json.gz` |
| H800 P0 | PREFILL | `/data/ycfeng/tmp/glm53_dsv4/profile04_results_1u00cy05/artifacts/case/runs/groundtruth_clean/dsv4-h800-p0-profile-20261024-04/torch_traces/prefill/dp0/dp0_pp0_tp0_dcp0_ep0_rank0.1792842834745705872.pt.trace.json.gz` |
| H800 P0 | DECODE | `.../dsv4-h800-p0-profile-20261024-04/torch_traces/decode/dp0/dp0_pp0_tp0_dcp0_ep0_rank0.1792842748090329211.pt.trace.json.gz` |

The H200 traces are the mainline evidence (D6). The H800 P0 traces are checked separately and only qualify the H800 case.

## 2. Streams

| Role | Created at | Used by | Stream id in the eager traces |
| --- | --- | --- | --- |
| main (the current stream of the forward) | — | everything not listed below | s23 |
| aux0 | `models/deepseek_v4/nvidia/model.py:1019-1023` (`aux_stream_list`, 3 streams shared by all layers) | outer projection: compressor `kv_score`; CSA: the indexer; HCA: the compressor | s35 |
| aux1 | same | outer projection: indexer `weights_proj`; CSA: the attention compressor | s39 |
| aux2 | same; passed to the indexer as its own `aux_stream` (`models/deepseek_v4/attention.py:285-298`) | outer projection: indexer compressor `kv_score`; inside the indexer: the indexer compressor | s43 |
| shared-expert stream | `utils/torch_utils.py:821-839` (`aux_stream()`, one process-wide stream), taken at `model_executor/layers/fused_moe/runner/shared_experts.py:69-70` | the MoE shared expert | s47 |

Graph replays place kernels on streams that the driver assigns, so a replay stream id says nothing about the capture stream. Concurrency inside a replay comes from the captured dependencies.

## 3. The stream graph of one layer, from the source

### 3.1 The two fork helpers

- `maybe_execute_in_parallel(fn0, fn1, ...)` (`utils/multi_stream_utils.py:16-62`): `fn0` on the current stream, `fn1` on the given stream. It runs both serially on the current stream when that stream is `None` or when a Breakable capture is active (`multi_stream_utils.py:45-49`). Host order: record the start event, issue `fn0`, then switch to the auxiliary stream, wait, issue `fn1`, record the done event; the current stream waits for the done event.
- `execute_in_parallel(default_fn, aux_fns, ..., enable)` (`multi_stream_utils.py:65-132`): the default function on the current stream, each non-`None` auxiliary function on its own stream. It runs serially only when the stream list is `None` or `enable` is false (`multi_stream_utils.py:105-108`; serial order: the default function first, then the auxiliary functions in order). **It has no Breakable check**, so it forks in every mode, PIECEWISE included. Host order when forking: record the start event, issue every auxiliary function on its stream (each waits for the start event and records its done event), then issue the default function; the current stream then waits for every done event (`multi_stream_utils.py:117-132`).

### 3.2 Capture kinds of the V2 runner

- Breakable capture is turned on for DeepSeek V4 at start-up (`config/vllm.py:1310-1337`; every retained `server.log` prints "Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1"). The V2 runner then uses it for PIECEWISE graphs (`v1/worker/gpu/cudagraph_utils.py:139-142`).
- **FULL**: captured with a plain `torch.cuda.graph(...)` and no Breakable capture (`cudagraph_utils.py:343-372`). So `maybe_execute_in_parallel` forks, `@eager_break_during_capture` functions run inline (no capture context, `compilation/breakable_cudagraph.py:96-103`), and `torch.cuda.is_current_stream_capturing()` is true.
- **PIECEWISE**: a Breakable capture (`breakable_cudagraph.py:381-382`). `maybe_execute_in_parallel` runs serially; `execute_in_parallel` still forks; each `@eager_break_during_capture` function ends the graph segment and runs eagerly on the capture stream, and is replayed in segment order.
- **NONE** (eager): no capture. Every fork is taken; the short-context branch of the indexer is possible.

### 3.3 Per-layer order

The V2 runner uses the narrow eager region: `_prepare_and_attn_fn = _prepare_and_attn` (`attention.py:302-307`; the wide region is only for the V1 runner). The attention forward (`attention.py:361-403`) is:

1. **Outer input projections** (`attention.py:515-575`), `execute_in_parallel` with `enable = rows <= VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` (default 1024, `envs.py:290, 2016-2017`):
   - main: `fused_wqa_wkv`;
   - aux0: compressor `kv_score` (CSA and HCA);
   - aux1: indexer `weights_proj` (CSA);
   - aux2: indexer compressor `kv_score` (CSA);
   - an uncompressed layer has no auxiliary function, so only the start event is recorded.
2. `fused_q_kv_rmsnorm` on main (`attention.py:382-388`).
3. **Attention input preparation** (`_prepare_and_attn`, `attention.py:433-506`):
   - CSA: `execute_in_parallel(main: wq_b + fused q-norm/rope/KV insert, [aux0: indexer, aux1: attention compressor], enable = aux streams exist)` (`attention.py:464-485`). Forks in every mode.
     - Inside the indexer, when the short-context branch is not taken: `maybe_execute_in_parallel(current = aux0: wq_b and q quant, aux2: indexer compressor)` (`attention.py:952-958`). Forks in FULL and NONE, serial under the PIECEWISE Breakable capture.
     - Short-context branch (`attention.py:906-930`): taken when `max_seq_len // 4 <= index_topk` (512) and the stream is not capturing, so only in NONE. The indexer compressor and `_fill_short_context_topk_indices` then run on aux0, the indexer returns `None`, and step 4 skips `indexer_op`.
   - HCA: `maybe_execute_in_parallel(main: wq_b + fused insert, aux0: compressor)` (`attention.py:486-494`). Forks in FULL and NONE, serial in PIECEWISE.
   - Uncompressed: serial on main.
4. **`_sparse_indexer_and_attn`** (`attention.py:577-602`), marked `@eager_break_during_capture`, all on main, no events: for CSA with an indexer output, `indexer_op` (index fill, `paged_mqa_logits`, top-k, global top-k), then `forward_mqa` (MLA metadata, sparse MLA, combine). It is an eager break in PIECEWISE and inline in FULL and NONE.
5. `_o_proj` on main.

Kernel names that identify the branches in the traces: HCA compressor `SparseAttnCompressC128Block8Kernel` and `SparseAttnNormRopeStoreKernel` (after `_save_partial_states_kernel`); CSA attention compressor `SparseAttnCompressNormRopeStoreC4Kernel`; CSA indexer compressor `_fused_kv_compress_norm_rope_insert_indexer_attn`; main-stream insert `fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel` (`...ReducedGrid` in prefill).

## 4. Stream graph by layer variant and mode

"Fork" means the work runs on its own stream after an event wait; "serial" means it runs on the current stream in host order. "Rows" is the attention layer's local row count.

| Layer variant | Region | FULL | PIECEWISE | NONE |
| --- | --- | --- | --- | --- |
| all | outer projections | fork if rows ≤ 1024, else serial | same | same |
| CSA | indexer (aux0) and attention compressor (aux1) beside the main insert | fork | fork | fork |
| CSA | inside the indexer: q path (aux0) and indexer compressor (aux2) | fork | serial on aux0 | fork; with short context: compressor and top-k fill only, on aux0 |
| CSA | `indexer_op` and `forward_mqa` | inline on main | eager break on main | on main; `indexer_op` skipped with short context |
| HCA | compressor (aux0) beside the main insert | fork | serial on main | fork |
| HCA | `forward_mqa` | inline on main | eager break on main | on main |
| uncompressed | everything | serial on main | serial, `forward_mqa` an eager break | serial on main |

## 5. Host issue order

The host order matters for eager forwards, where the device can only overlap work the host has already issued. In a graph replay only the captured dependencies matter.

- `execute_in_parallel`: auxiliary functions first, in list order, then the default function (observed: in a CSA layer the aux0, aux1 and aux2 kernels are issued before `fused_wqa_wkv` and the main insert).
- `maybe_execute_in_parallel`: `fn0` (current stream) first, then `fn1` (observed: in an HCA layer the main insert is issued before the aux0 compressor).
- Shared expert: issued after the routed path (§7).

## 6. Trace checks

Scripts and outputs are in [evidence/stream_graph/](evidence/stream_graph/) (§10). All counts are per forward of TP rank 0 of DP lane 0.

### 6.1 Eager forwards (NONE): kernels per stream

`peek_sequence.py` over the P2-32 prefill trace (`sequence_p2_32_prefill_dp0.txt`) and the H800 P0 prefill trace (`sequence_h800_p0_prefill_dp0.txt`):

| Forward | s23 (main) | s35 (aux0) | s39 (aux1) | s43 (aux2) | s47 (shared expert) | Source reading |
| --- | --- | --- | --- | --- | --- | --- |
| Real prefill, 6144 rows | 2172 | 123 = 21 × 3 + 20 × 3 | 42 = 21 × 2 | 42 = 21 × 2 | 0 | Rows > 1024: outer projections serial. CSA: q path (3 kernels) on aux0, indexer compressor (2) on aux2, attention compressor (2) on aux1. HCA: compressor (3) on aux0. Rows > 256: shared expert on main |
| Dummy, 1 row | 1675 | 205 = 21 × 5 + 20 × 5 | 84 = 21 × 4 | 42 = 21 × 2 | 215 = 43 × 5 | Rows ≤ 1024: outer projections fork (2 kernels each on aux0, aux1, aux2). Short context: indexer compressor and top-k fill (3) on aux0, so aux2 has only its outer projection. Rows ≤ 256: shared expert (5 kernels) on its stream in every layer |

The H800 P0 counts are the same for both kinds of forward. Some H800 real forwards have 2086 kernels on main instead of 2172. `dispatch_kernel_diff.py` (`kernel_diff_h800_p0_prefill_160_256.txt`) shows the difference is only in collective kernels: `AllGatherV` × 129 and `Reduce` × 43 in the 2172 forward against `AllGather` × 87 and `ReduceScatter` × 86 in the 2086 forward. Both have 6144 rows (the first `fused_q_kv_rmsnorm` grid is `[6144, 2, 1]`). This is the difference between unequal and equal sequence-parallel sizes across DP lanes, not a stream difference.

### 6.2 PIECEWISE decode: the eager breaks

`piecewise_breaks.py` (`piecewise_breaks_p2_32_decode_dp0.txt`, `piecewise_breaks_h800_p0_decode_dp0.txt`): every PIECEWISE forward has 44 graph launches and 174 eager kernels, all on s23. 174 = 43 × 2 (sparse MLA and combine of every layer) + 21 × 4 (index fill, `paged_mqa_logits`, top-k, global top-k of every CSA layer) + 4 MLA metadata kernels. The 44 graph segments are the 43 breaks plus one. This matches §3.3 step 4 and is identical on H200 and H800.

### 6.3 Graph replays: concurrency of the forked work

`kernel_overlap.py` measures, per mode, the device time of a named kernel set and the part of it during which another kernel of the same forward also ran ("overlapped"). Decode traces, FULL and PIECEWISE forwards:

| Kernel set | Source rule | P2-32 FULL / PW | P2-64 FULL / PW | H800 P0 FULL / PW |
| --- | --- | --- | --- | --- |
| HCA compressor (40 kernels per forward) | `maybe_execute_in_parallel`: fork in FULL, serial in PIECEWISE | 95.1% / 0.0% | 94.7% / 0.0% | 96.4% / 0.0% |
| CSA attention compressor (21 kernels per forward) | `execute_in_parallel`: fork in both | 69.7% / 71.6% | 74.5% / 73.4% | 70.3% / 72.6% |

Both results follow the source rules of §4 on all three cases. Forwards counted: FULL 117 / 125 / 110, PIECEWISE 3 / 4 / 5.

## 7. Finding: the MoE shared expert runs on its own stream

This is outside the attention scope of design §3.2, which composes the shared expert serially.

**Source rule.**

- Overlap condition (`shared_experts.py:99-119`): the shared expert runs on its stream (`MULTI_STREAM_OVERLAPPED`) when the platform is CUDA, the stream exists and the MoE input rows are at most `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD` (default 256, `envs.py:289, 2005-2006`). It does not when overlap is disabled (`shared_experts.py:79-97`: sequence-parallel sharding of the layer, EPLB with an all-to-all backend outside the overlap-safe list, or FlashInfer NVL two-sided kernels) or when the modular kernel overlaps it internally. Otherwise it runs on main (`NO_OVERLAP`). The rule has no Breakable check, so it applies in FULL, PIECEWISE and NONE.
- Order (`moe_runner.py:852-884, 583-618`; `shared_experts.py:121-151`): at MoE entry, before the gate, the shared-expert stream waits for main. Main then runs the gate, the dispatch, the routed experts and the combine (`forward_modular`). After that the host issues the shared expert on its stream, and main waits for it right away, before `_maybe_combine` adds the outputs. With `NO_OVERLAP` the shared expert runs on main before routing.
- So in a graph replay the shared expert depends only on the main-stream point before the gate, and it can run beside the gate, the EP dispatch, the routed experts and the EP combine. In an eager forward it is issued after the routed path, so overlap depends on how far the host runs ahead of the device.

**Trace check.** Kernels matched: the two shared-expert GEMMs (`swapAB<4096u, 4096u` and `swapAB<4096u, 2048u`) and `act_and_mul_kernel`, 129 = 43 × 3 per forward. The dummy sequence in §6.1 shows the same five shared-expert kernels per layer on s47, issued after the routed path's combine.

| Case | FULL: overlapped / busy per forward | FULL share | PIECEWISE share | Median FULL forward device span | Overlapped / span |
| --- | --- | --- | --- | --- | --- |
| H200 P2-32 | 479.1 / 743.1 µs | 64.5% | 54.0% | 17.9 ms | 2.7% |
| H200 P2-64 | 469.0 / 754.6 µs | 62.1% | 53.6% | 24.1 ms | 1.9% |
| H800 P0 | 533.8 / 820.0 µs | 65.1% | 54.5% | 19.2 ms | 2.8% |

**Inference.** A serial shared expert over-predicts a FULL decode forward by at most the overlapped time, about 1.9-2.8% of the forward. The real saving is smaller, because concurrent kernels share the device. This is material next to the A4 and A6 limits (median 5%), so it is recorded as material.

**Recommendation (adopted by the lead under D-NO-INTERRUPT).** Keep entry 2 as designed (attention layer only). Model the shared-expert fork as a later explicit change after entry 2: it spans the EP dispatch and combine that `expert_parallel.py` prices, so it needs its own design note. A design revision 2e or a Codex round 4 for it needs YC's approval.

## 8. What entry 2 takes from S1

The composition selects the per-call graph of §4 from four inputs. None of them is a literal in Frontier:

1. **Rows against the multi-stream threshold**: the layer's local rows against `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD`, read from the runtime identity the producer records with the attention rows (design §3.2).
2. **Capture kind**: FULL, PIECEWISE (Breakable) or NONE, from the dispatched descriptor (entries 20-25).
3. **Short-context state** (NONE only): `max_seq_len // compress_ratio <= index_topk`, with `index_topk` from the model config.
4. **Layer variant**: uncompressed, CSA or HCA, from the compress ratio of the layer.

Node weights are the predicted scope costs plus the gaps of design §3.3. The joins are the event waits of §3.1. The shared expert is not in this graph (§7).

## 9. Corrections to the design wording

S1 refines two sentences of [design.md](design.md). The approach does not change, so this is not a design revision:

- §3.2 says "nested branches run serially under a Breakable (PIECEWISE) capture". This holds for the `maybe_execute_in_parallel` forks (the HCA compressor and the q path / indexer compressor fork inside the CSA indexer). It does not hold for the `execute_in_parallel` forks (the outer projections and the CSA indexer / attention compressor fork), which fork in PIECEWISE too (§6.3: 71.6-73.4% overlap in PIECEWISE).
- §5 stage 1a says "under a Breakable capture the compressor runs on the current stream instead of the auxiliary one". This holds for the HCA compressor and the CSA indexer compressor, not for the CSA attention compressor.

## 10. Evidence files

In `h3o_operator_costs/evidence/stream_graph/` (copied from `/data/ycfeng/tmp/glm53_dsv4/h3o_b/s1/`, some renamed on copy). Python: `/data/ycfeng/envs/frontier-py310/bin/python` 3.10.6, `ulimit -v 31457280`.

| File | Command | Content |
| --- | --- | --- |
| `peek_sequence.py` | `python peek_sequence.py <trace> <limit>` | Per eager dispatch: real or dummy, kernels per stream, runtime-call counts, then the first `<limit>` launches in host order with their streams |
| `sequence_p2_32_prefill_dp0.txt` | `peek_sequence.py <P2-32 prefill trace> 1400` | §5, §6.1 (H200) |
| `sequence_h800_p0_prefill_dp0.txt` | `peek_sequence.py <H800 P0 prefill trace> 0` | §6.1 (H800) |
| `piecewise_breaks.py` | `python piecewise_breaks.py <trace>` | Eager kernels inside PIECEWISE dispatches |
| `piecewise_breaks_p2_32_decode_dp0.txt`, `piecewise_breaks_h800_p0_decode_dp0.txt` | over the P2-32 and H800 P0 decode traces | §6.2 |
| `dispatch_kernel_diff.py` | `python dispatch_kernel_diff.py <trace> <A> <B>` | Kernel-name count differences between two eager dispatches |
| `kernel_diff_h800_p0_prefill_160_256.txt` | over the H800 P0 prefill trace, dispatches 160 and 256 | §6.1 |
| `kernel_overlap.py`, `run_overlap.sh` | `bash run_overlap.sh` (9 runs) | Overlap of a kernel set per mode |
| `overlap_{shared_expert,hca_compressor,csa_attn_compressor}_{p2_32,p2_64,h800_p0}_decode_dp0.txt` | from `run_overlap.sh` | §6.3, §7 |

## 11. Limits

- Only TP rank 0 of DP lane 0 was checked in each case.
- Eager forwards with 2-1024 rows were not in the traces (real prefills have 6144 rows, dummies 1). The fork rule for that range comes from the source only.
- The short-context branch was seen only in dummy forwards.
- PIECEWISE results rest on 3-5 forwards per trace.
- "Overlapped" measures concurrency, not time saved.
- Replay stream ids carry no meaning (§2); the replay checks use concurrency only. Graph-node ownership is stage 1a's work.
- No GPU run; 0 GPUs used.
