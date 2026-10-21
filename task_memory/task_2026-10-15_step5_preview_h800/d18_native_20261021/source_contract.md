# D18 Step native runtime source contract

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded source-derived candidate limits, streaming schema and native graph evidence route. |

## Source and identity

Read-only exported source root: `/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm`. Existing image source manifest: primary task `native_pd_20261020/image_source/sha256_in_image.txt`. Locked native digest remains `sha256:69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20` and installed version `0.1.dev22912+g8224bb82f`. The stage binds the additional inspected files and the prior native pins; the unchanged worker must verify their installed hashes before serving. Reading exported source alone is not a new installed-image observation.

## Streaming and exact inputs

- `entrypoints/openai/completion/protocol.py:169-177` declares `return_token_ids`, prompt IDs once, and generated token-ID deltas per streamed chunk.
- `entrypoints/openai/completion/serving.py:337-375,404-421` distinguishes echo from normal generated deltas and places both `prompt_token_ids` and generated `token_ids` **inside the choice**. The client requests `echo=false`, `add_special_tokens=false`, `return_token_ids=true`; no text tokenization is used. The first/last nonempty generated ID delta supplies client TTFT/E2E endpoints, even when native text is empty.
- The final usage event is independent of the last output token. It corroborates 6144 prompt / 1024 output counts but does not extend E2E or the throughput denominator. Prompt count plus returned-ID SHA-256 must equal the original deterministic prompt.
- Vocabulary is `text_config.vocab_size=128896`, from unchanged original metadata. Prompts use the existing `vllm_burst_driver.prompt_token_ids` and stable common template IDs, not repeat-qualified transport IDs.
- `engine/arg_utils.py:707,2071` keeps `speculative_config=None` without a speculative configuration. No speculative flag is passed; the metadata's three draft layers do not enable MTP. Fixed token-ID completions use no chat template, reasoning parser or thinking option.

## Unchunked limits and topology

- Context candidate **7168** is 6144+1024, not a measured capacity result.
- `config/scheduler.py:314-332` rejects unchunked `max_num_batched_tokens < max_model_len`, and token budget below sequence capacity. **8192 >= 7168 >= 6144** and **8192 >= 128** satisfy the source constraints. 8192 accommodates one full prompt with concurrent decode tokens; it is not a promise that H200 memory accommodates every possible 128-sequence state.
- **128 sequences per DP lane** (`config/scheduler.py:62-102`) allows the entire 128-request formal population to remain admitted even if lane occupancy is uneven. It is a candidate cap, not a claim of 128 simultaneously resident requests per lane or saturation.
- Each role is TP8/DP4/EP32/PP1; one TP8 DP lane per 8-GPU node, eight nodes / **64 H200**, `step_main`, non-preemptible. No forced cache-block override. Native FP8 metadata/exclusions are copied unchanged; no forced expert routing. Chunked prefill and prefix caching are explicitly disabled.

## Graph selection and runtime evidence

- `config/vllm.py:100-123,770-776,814-823` includes Step in CUDA's default breakable-graph architectures. Do not copy GLM compilation flags or force `VLLM_USE_BREAKABLE_CUDAGRAPH`.
- `config/compilation.py:610-641,873-878` supports the named `FULL_AND_PIECEWISE` mode: full decode and piecewise mixed/prefill where capture sizes cover them.
- `config/vllm.py:2381-2391,2470-2526`: with no MTP, max sequences 128 and H200's default ceiling512, the native ceiling is `min(2*128,512)=256`. The stage explicitly freezes that ceiling, preserving native capture-size inference: 1,2,4, multiples of8 through248, then256. The 6144-token P prefill is above this ceiling and is **not** made graph-covered by this configuration; reached D tail/mixed batches may be piecewise, and uniform decode is the full-graph path. Do not relabel P prefill as graph compute.
- Step's sparse backend declares `UNIFORM_SINGLE_TOKEN_DECODE` in `models/step5/nvidia/sparse_attention.py:213-223`. `v1/worker/gpu_model_runner.py:4021-4060` re-dispatches after DP synchronization and reports the resulting runtime mode. Configuration alone cannot prove FULL survives that coordination.
- The existing native `--cudagraph-metrics` (`engine/arg_utils.py:1661`, `config/observability.py:82-84`) records padded/unpadded tokens, actual dispatch modes and frequencies without tensor readback or CUDA fences. `v1/metrics/loggers.py:129-134` and `compilation/cuda_graph.py:40-124` write the native table. Both qualification and canonical runs enable it; native host/logging overhead remains included, with no correction. This is native observability, not R2 bootstrap or content capture.
- `compilation/cuda_graph.py:235-265` routes matching runtime modes into the CUDA graph wrapper; the cached entry's `cudagraph.replay()` at line361 is the native replay route. `v1/metrics/loggers.py:338` emits the native dispatch table. Parent must inspect each lane's native capture/startup and repeated FULL decode dispatch evidence, not infer replay from a successful process or sixteen aggregate warmups.

## Still required from the parent

For each stage: platform terminal receipt and creator/resources, `nodes/*/result.json`, `nodes/*/server.log`, `replay/canonical_summary.json`, `replay/client_requests.jsonl`, and `replay/proxy_requests.jsonl`, under that stage's exact cloud output. These settle installed-source identity, 64 distinct H200 devices, TP/DP/EP roles, resolved limits/cache capacity/graph policy, per-lane native dispatch frequencies, accepted lengths, transfer split/failures and phase idle boundaries. Missing rank-level evidence remains unverified; aggregate native tables are not a fabricated per-TP-rank replay census. Parent-owned R2 content/lifetime judgment is separate and unchanged.
