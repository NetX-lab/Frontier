## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Qualified fresh linear and attention profile rows against the first formal prefill scopes, preserving DP/rank identity and recording concrete boundary exclusions. |

# First Formal Prefill Profile Qualification

This is a read-only shape and timed-boundary qualification. It reports no timing averages, operator gaps, TTFT attribution, or repair recommendation. Only the new H200 profiles are eligible as numerical inputs. MoE is outside this document's scope.

## Evidence and identity

Paths below are relative to the active worktree unless explicitly absolute. `TASK` is `task_memory/task_2026-09-07_issue26_ttft_h200`.

- Formal identity: `TASK/analysis/formal-operators-03/receipt.json` and its per-worker `raw_selected_ops` paths.
- Audited scope definitions: `TASK/analysis/operator_scope_contract.md`.
- `LINEAR`: `TASK/runs/h200-fresh-profiles-02/runtime/profiles/compute/h200/qwen3-a3b-30b-moe/linear_op.csv`.
- `ATTENTION`: `TASK/runs/h200-fresh-profiles-02/runtime/profiles/compute/h200/qwen3-a3b-30b-moe/attention_combined.csv`.
- Profile launch: `tests/performance/issue26_h200_fresh_profiles_worker.sh:15`, BF16, CUDA_EVENT, attention TP4, FLASHINFER, block size 16, maximum model length 16384.
- Runtime source prefix `VLLM`: `/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908`.

Each identity below is a separate comparison target. The profile table applies separately to each; it does not pool TP ranks or pair batches across DP lanes.

| DP | TP | PP | First formal batch | Client request | Requests / prefill tokens / decode tokens |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 3851 | `pf4096_dc1024:0` | 1 / 4096 / 0 |
| 0 | 1 | 0 | 3851 | `pf4096_dc1024:0` | 1 / 4096 / 0 |
| 0 | 2 | 0 | 3851 | `pf4096_dc1024:0` | 1 / 4096 / 0 |
| 0 | 3 | 0 | 3851 | `pf4096_dc1024:0` | 1 / 4096 / 0 |
| 1 | 0 | 0 | 3863 | `pf4096_dc1024:2` | 1 / 4096 / 0 |
| 1 | 1 | 0 | 3863 | `pf4096_dc1024:2` | 1 / 4096 / 0 |
| 1 | 2 | 0 | 3863 | `pf4096_dc1024:2` | 1 / 4096 / 0 |
| 1 | 3 | 0 | 3863 | `pf4096_dc1024:2` | 1 / 4096 / 0 |

Both CSVs omit an explicit `head_dim` column. The profile model JSON supplies `head_dim=128`, hidden size 2048, 32 Q heads and 4 KV heads (`data/config/models/qwen3-a3b-30b-moe.json:10`). The linear producer uses `config.get_head_dim()`; the attention base uses `model_config.get_head_size()` (`frontier/profiling/attention/backends/base_attention_wrapper.py:51`). Thus TP4 gives 8 local Q heads, 1 local KV head, and head dimension 128; deriving 64 from 2048/32 would be wrong for this model. This is config/producer evidence, not a field observed in the CSV or runtime metadata.

## Exact profile coordinates

Physical CSV line numbers count the header as line 1; data indices are zero-based.

| Source and coordinates | Exact metadata and populated scopes | Qualification |
| --- | --- | --- |
| `LINEAR`, line 51, data index 49 | `num_tokens=4096`, TP4, BF16, CUDA_EVENT, generic, quantization none; Q/KV heads 32/4, hidden size 2048, `use_qk_norm=True`. Only `attn_pre_proj`, `attn_rope`, `attn_post_proj` timing families are populated. | Exact logical shape for the three attention linear scopes; boundary rules below determine direct eligibility. No norm values exist on this TP4 row. |
| `LINEAR`, line 22, data index 20 | Same token count, hidden size, precision and measurement family; TP1. `typed_operator_contracts` explicitly classifies embedding/input norm/post-attention norm as `tensor_parallel_mode=replicated`, selected TP1. Only these three timing families are populated. | Legitimate replicated local norm profile for the TP4 serving case, subject to residual-boundary rules below. It is not a sharded attention TP1 substitute. |
| `ATTENTION`, line 68, data index 66 | TP4, BF16, CUDA_EVENT, FLASHINFER, block size 16, model length 16384; batch size 1, `seq_lens=[4096]`, prefill chunk 4096, prior KV 0, total tokens 4096; pure prefill, non-mixed, non-chunked, chunk range 0..4096, generic, quantization none. | Exact logical shape for cache save and prefill. |
| `ATTENTION`, line 82, data index 80 | Every non-`time_stats.*` field equals line 68, including the empty true-mixed extension fields. | A second exact candidate, not a distinct shape. Retain both row identities; this audit does not select one, average them, or assert that the timing samples are identical. |

All timing columns use the producer's `time_stats.<scope>.<stat>` naming. Populated attention `attn_decode` timing fields do not make these pure-prefill rows eligible decode evidence: the producer enters that timer even when the decode branch has no work.

## Per-operator timed-boundary eligibility

Here, eligible means the logical input shape and included operation family support a comparison after the actual Frontier batch is aligned. It does not assert identical tensor strides, kernel choices, scheduling context, or measured latency.

| vLLM scope | Profile coordinate / field family | Direct eligibility | Inspected boundary and next-use rule |
| --- | --- | --- | --- |
| `attn_pre_proj` | `LINEAR:51`, `time_stats.attn_pre_proj.*` | Eligible | Both scopes include QKV projection, Q/K/V split, and Q/K normalization. Profile `frontier/profiling/linear_op/linear_op_impl.py:500`; runtime `VLLM/vllm/model_executor/models/qwen3_moe.py:306`. Compare the parent once. |
| `attn_rope` | `LINEAR:51`, `time_stats.attn_rope.*` | Eligible | Both enclose rotary embedding with the configured head dimension. Profile timer is constructed at `linear_op_impl.py:492`; runtime scope is `qwen3_moe.py:361`. |
| `attn_post_proj` | `LINEAR:51`, `time_stats.attn_post_proj.*` | Ineligible as a direct same-name comparison | Profile `RowParallelLinear(..., reduce_results=False)` at `linear_op_impl.py:473` measures local projection. Runtime parent at `qwen3_moe.py:382` includes TP4 reduction. Follow the existing contract's `attention_post_proj_time + attention_all_reduce_time` group when Frontier rows become available. The unselected runtime child cannot be recovered by naming assumptions. |
| `attn_kv_cache_save` | `ATTENTION:68` and `:82`, `time_stats.attn_kv_cache_save.*` | Eligible at the scope boundary | Both wrap `reshape_and_cache_flash` with cache slices, slot mapping, KV dtype and K/V scales. Profile `frontier/profiling/attention/backends/flashinfer_attention_wrapper.py:400`; runtime `VLLM/vllm/v1/attention/backends/flashinfer.py:861`. Preserve stride/layout limitations below. |
| `attn_prefill` | `ATTENTION:68` and `:82`, `time_stats.attn_prefill.*` | Eligible at the scope boundary; duplicate row identity remains explicit | Both time paged prefill `wrapper.run`, pass K/V scales and a preallocated `out=` slice, and exclude planning, cache write and output allocation. Profile `flashinfer_attention_wrapper.py:420`; runtime `flashinfer.py:913`. Runtime TRT-LLM attention requires SM100 (`VLLM/vllm/utils/flashinfer.py:157`), excluding that alternate branch on the verified H200 hardware. |
| `input_layernorm`, runtime layer 0 | `LINEAR:22`, `time_stats.input_layernorm.*` | Ineligible | Runtime first layer has no residual and uses standalone RMSNorm. Profile `GPTModel.forward` resets `residual=hidden_states` before every measured block (`linear_op_impl.py:1088`), so its input norm is always the fused residual/RMSNorm branch. No standalone first-layer row was identified. |
| `input_layernorm`, runtime layers 1–47 | `LINEAR:22`, `time_stats.input_layernorm.*` | Eligible at the fused parent boundary | Both execute residual-add/RMSNorm, using replicated full hidden width 2048. Profile branch `linear_op_impl.py:908`; runtime branch `qwen3_moe.py:453`. This is not norm-only or add-only evidence. |
| `post_attention_layernorm` | `LINEAR:22`, `time_stats.post_attention_layernorm.*` | Eligible at the fused parent boundary | Both execute residual-add/RMSNorm; profile `linear_op_impl.py:939`, runtime `qwen3_moe.py:552`. Count the parent once; do not add its nested runtime `add` duration. |
| `attn_input_reshape`, `attn_output_reshape`, embedding | Attention reshape fields; `LINEAR:22` embedding field | Ineligible as direct default-scope comparisons | These are profile fields without matching selected default compute scopes in the formal runtime record. The runtime singleton `tensor_parallel_allreduce` is embedding communication, not embedding lookup. |
| `attn_decode` | `ATTENTION:68` and `:82` decode timing fields | Ineligible for this batch | The formal local batch contains zero decode tokens. The profiling timer surrounds a conditional branch that does not call the decode kernel for this row. |

## Limits retained for the upcoming comparison

Runtime metadata was disabled, so the selected formal records do not establish exact per-call strides, cache layout or kernel implementation. The profile attention forward explicitly makes Q/K/V contiguous before the cache timer (`flashinfer_attention_wrapper.py:393`); source-level timed-boundary equality therefore cannot establish identical cache-write input strides. Do not promote the eligible rows into exact kernel-input equivalence without evidence.

The repeated standalone CUDA_EVENT profile and the instrumented runtime CUDA_EVENT parent scopes have different execution contexts. Runtime events measure ordered GPU work and waits, including possible device idle time while the host queues work, as documented in `operator_scope_contract.md`. Qualification does not turn an operator timing difference into a CPU overhead measurement.

No predictor, configuration, profiling data, or runtime code changed. The actual fresh Frontier03 batch/ledger and the duplicate attention-row handling remain inputs to the later numerical operator comparison. This audit does not satisfy a multi-batch operator-gap gate by itself.
