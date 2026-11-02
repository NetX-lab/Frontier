# Native FULL dummy producer and launch requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-21 | Documented source-backed semantics, isolated output schema, and the GPU/kernel-only admission limits. |

## Native evidence

All references below are git objects at
`2cf0a6915ce544dc493a0990f2ea38d81601128a`, not the current native working tree.
Repository: `/data/ycfeng/Frontier/.real-engine/vLLM-BS`.

| Native path and lines | Contract reused |
| --- | --- |
| `vllm/v1/worker/gpu_model_runner.py:5940-6059` | `_dummy_run(1, uniform_decode=True)` retains one actual query and request; native uniform query length is one without speculation. |
| `vllm/v1/worker/gpu_model_runner.py:6077-6190` | The batch descriptor supplies padded tokens/requests. Native slots are filled with -1, sequence lengths are `[1,0,...]`, query starts are `[0,1,1,...]`, and FULL restages capture metadata. |
| `vllm/v1/worker/gpu_model_runner.py:4055-4150` | Native DP coordination chooses a shared width and then re-dispatches the graph descriptor. The producer substitutes only that descriptor on its single reduced-model GPU; it does not emulate or time the collective. |
| `vllm/v1/worker/gpu_model_runner.py:4205-4277` | Native per-cache-group slot buffers and per-layer aliases are reused. |
| `vllm/v1/worker/gpu_model_runner.py:5554-5574` | FULL wraps the complete native model in `CUDAGraphWrapper`. |
| `vllm/v1/worker/gpu_model_runner.py:6949-7177` | Native `capture_model`, `_warmup_and_capture`, and `_capture_cudagraphs` own capture stream, pool, warmup and dispatcher keys. |
| `vllm/compilation/cuda_graph.py:240-361` | Capture calls the runnable once; replay executes the stored CUDA graph, not Python module hooks. |
| `vllm/v1/cudagraph_dispatcher.py:133-157,205-231` | Uniform FULL request padding and legal FULL_DECODE_ONLY capture sizes are selected natively; no default width-one assumption. |

The persistent source-contract test reads these exact git objects. It is optional
outside hosts with a native source checkout and enabled by
`FRONTIER_VLLM_BS_ROOT`. It is not a GPU execution test.

## What the producer actually measures

`frontier.profiling.attention.dsa_full_dummy` reuses the DSA reduced model and
module inventory. The native startup graph is discarded locally and recaptured
with external timing-event nodes around each native Indexer and MLAAttention
scope. Native capture initially uses its full uniform bucket; actual-query-one
replays restage native padded dummy buffers before executing that same graph.
The hooks must not execute at replay; missing scopes, recapture, invalid slot or
padding witnesses, or missing metadata restaging fail explicitly.

Compilation is disabled (`CompilationMode.NONE`) to retain these Python scope
boundaries during native graph capture; CUDA graphs remain enabled separately.
The reduced-model convention follows the existing DSA producer. Equality of
these scope kernels with the formal compiled native model still requires the
primary's selected-kernel qualification; it is not established by CPU tests.

The interval between captured external events is **CUDA_EVENT GPU wall time**.
It includes device gaps within the scope, not Python overhead before replay.
It is not the sum of individual kernel durations. No KERNEL_ONLY values or
kernel counts are emitted, and no regular serving measurement is borrowed.

## Exact output schema and coverage mapping

The output is a new JSON file opened exclusively, schema `dsa_full_dummy.v1`.
It is deliberately not an `attention.csv`/`dsa_attention.csv` file.

- Root: `schema_version`, `native_revision`, `model_config_revision`,
  `model_config_sha256`, `plan` (existing reduced-model plan, including shape and
  HF overrides), `capture_sizes`, `measurement_type="CUDA_EVENT"`,
  `kernel_only_qualified=false`, `consumer_admission=false`,
  `gpu_execution_observed` (false for plan; true only after successful run).
- Measured-run additions: `identity` from existing `RuntimeIdentity` (runtime
  versions, device/capability, source/config, quantization, dummy weight source,
  target TP / physical reduced-engine TP, FULL graph mode), `inventory`
  (selected backend/implementation and native layer index-reuse inventory),
  `kv_cache_dtype`, `resolved_capture_sizes`, and `records`.
- Each record: `schema_version`, `measurement_type="CUDA_EVENT"`,
  `execution_kind="full_dummy"`, `graph_mode="FULL"`, `global_layer_id`,
  `attention_variant`, `scope`, `repeat` (zero-based measured replay),
  `time_ms`, and `witness`.
- Witness: `capture_width`, `actual_query_tokens=1`, `actual_requests=1`,
  `cached_context_tokens=0`, every cache group's `slot_mappings` (`-1` at every
  padded token), `seq_lens=[1,0,...]`, `query_start_loc=[0,1,1,...]`, and native
  scalar `metadata` keyed by layer name. Metadata scalars are observed, not
  synthesized assumptions about padded `num_actual_tokens`.

For eight reduced GLM layers there are four full-index layers and four
shared-index layers. Each measured replay yields 12 records: eight MLA plus
four indexer scopes. With `S` widths and `R` measured replays, expect `12*S*R`
records. Shared-index has no indexer scope, not an invented zero measurement.
Coverage is **only** query-one/context-zero at each exact requested capture
width and the recorded runtime/model geometry. Nothing covers real serving,
another context, an intermediate graph width, or another device.

## Offline plan and root-only GPU recipe

CPU example (the output path must not already exist):

```bash
PYTHONPATH=. /data/ycfeng/envs/frontier-py310/bin/python \
  -m frontier.profiling.attention.dsa_full_dummy plan \
  --model_name zai-org/GLM-5.3 \
  --model_config_path data/config/models/zai-org__GLM-5.3.json \
  --target_tensor_parallel_size 8 --block_size 64 \
  --max_context_len 7168 --max_kv_tokens_per_lane 14336 \
  --max_num_seqs 8 --chunk_budgets 1024 --capture_sizes 2 4 8 \
  --source_revision 2cf0a6915ce544dc493a0990f2ea38d81601128a \
  --model_config_revision <CONFIG_REVISION> \
  --expected_attention_backend FLASHMLA_SPARSE \
  --output <NEW_LOCAL_PLAN_JSON>
```

The numerical limits and widths in this CPU example are **not frozen native
serving settings**. `chunk_budgets` and KV capacity belong to the reused ordinary
planner; its cases are included for provenance but are not executed here. The
GPU producer disables chunking and uses `max_num_batched_tokens=max_context_len`.

After H200 qualification, the root can use the same command with `run`, the
**final native capture sizes and max_num_seqs/max_model_len**, observed lane KV
capacity, and explicit `--model_dir <MATCHING_CONFIG_DIRECTORY>`
`--gpu_memory_utilization <APPROVED_VALUE>` `--seed <SEED>`.
Use worker Python with locked native vLLM and compiled dependencies, not the CPU
test environment. Native source identity is supplied by the trusted launch
package and must be independently verified there; the producer checks config
bytes and expected revision argument, not an installed source checkout's hash.

Resource requirement: one H200 reduced-model worker, not 64 GPUs. This measures
rank-local attention scopes, not TP/DP/EP communication or formal 128-request
serving. Mount only the root-approved frozen Frontier/native/model inputs;
retain measured JSON under `/mnt/codesign-exp/ycfeng/...`. Submission and
monitoring belong to the primary; this slice performs neither.

Root qualification must inspect native selected backend, effective FULL graph
mode and sizes, actual device and versions, all expected scopes and witnesses,
repeat stability, lack of recapture/Python replay hooks, numerical event values,
and terminal status. Native event support and recapture behavior have not yet
been executed on a GPU. No measured JSON/CSV is supplied by this slice.

## Bounded consumer and kernel-only follow-up (not implemented)

1. Preserve the current `DSACoverageError` at
   `frontier/execution_time_predictor/dsa_predictor.py:154-160` and caller
   `sklearn_execution_time_predictor.py:7694-7706` until method-correct measured
   data and explicit admission approval exist.
2. The existing kernel-only trace parser uses Python `record_function` ranges
   around individual kernel-launch APIs. On whole-model replay it sees one
   graph launch for many layers, so those eager scope ranges do not attribute
   replay kernels. Do not label captured-event wall intervals KERNEL_ONLY.
3. A future kernel-only producer needs independently qualified graph-node-to-
   scope mapping and replay kernel durations, for example CUPTI graph-node
   creation callbacks tied to capture-time scope ownership. Measure graph replay
   kernels, not capture warmups. This is a separate profiling instrumentation
   capability; no such mapping has been observed in this CPU-only slice.
4. After that qualification, propose the narrow existing DSA loader/predictor
   admission: isolated full-dummy schema, exact runtime/config/variant/method/
   width keys, measured scope aggregation across sampled layers/repeats, no
   interpolation into ordinary serving bounds. Consumer and metadata-interface
   changes require primary review/approval and are outside this ownership.

Consequently, Q11's native FULL CUDA-event producer has source/CPU evidence,
but full **kernel-only producer and end-to-end coverage unblocking remain open**.
