# W128 native stream probe requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Root allocated two independently reviewed P2 repairs within275lines; existing root image receipts remain original-source evidence. |
| 2026-10-24 | Recorded root's bounded source-only probe allocation. |

[Original Request] Implement W128-NATIVE-STREAM-PROBE, owning only new `tests/comparison/calibration/native_profile_probe.py` (at most250 gross lines) and this record directory. No GPU/platform/network/Docker/Git writes; root handles packaging, reviews and submission later. Root-owned collector repair SHA256 `17d6dea11d35a2dc9849a11986fe5df46b45adb84d2d2bdeb958642a851b26ef` is pending review.

[Original Request] Actually import and invoke `ServingProfile.stream/trace_stream` on Torch default and auxiliary CUDA streams, before/during two native profiler cycles. Use the actual `vllm.profiler.wrapper.TorchProfilerWrapper` and `ProfilerConfig`, reusing one wrapper and underlying profiler. Every eager target needs same-thread CPU annotation -> runtime correlation -> kernels on the directly queried CUPTI stream. Retain original native traces, exceptions, runtime/source/library/GPU identities and all priming evidence.

[Original Request] Independently capture two uniquely annotated tiny CUDA operations while the first native profiler cycle is active, instantiate/replay, stop/start the same wrapper, and replay in cycle2. Preserve graph/node IDs and captured annotations. Only directly proven exact graph/node identities may establish per-op correspondence; otherwise UNKNOWN without invalidating eager results. No `get_graph_data`, custom callbacks, retained-topology APIs, numeric ID remapping, source/native edits or paused GLM work.

[Decision] Root's eventual run uses1H800/codesign/non-preemptible,16CPU,64GB,600seconds, image `hub.i.basemind.com/vllm-openai/vllm-openai@sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635`, existing CUDA13.0 compat configuration and Torch-preloaded CUPTI SONAME. No new path flags, fallbacks or symlinks. CLI has only fresh `--output-dir`; launch config owns its approved worker-local/personal-cloud location and the existing `FRONTIER_NATIVE_PROFILE_LIMITS` environment variable.

[Decision] This is probe-source delivery, not GPU stream/capture acceptance or a model/serving cost claim. CPU verification is AST, `--help`, fresh imports without CUDA; no persistent mocked tests or launcher. Required matrix results belong to root's source/package acceptance, not a claimed GPU proof here.

[Original Request — repair follow-up] Repair reviewer `ses_e962e4b01ffeqUa7O9uUTzGqSe`'s two P2 findings: require actual successful `capture_returned` and both `replay_returned` flags before graph PASS; compute the eager verdict before optional priming/graph parsing and isolate graph schema exceptions. Same single source file plus this directory, at most275gross lines, no compression, Git writes, GPU/platform/network/Docker operation or delegation. Preserve original source SHA256 `237a82f1176a6acefff82b0e329ee09679b602383f4045b4c4751f1df85759d7` and the original CPU errors.

[Decision] First reproduce RED using original bytes, then run identical retained temporary CPU fault controls against repaired real functions. Valid exact graph/eager positives must remain PASS; incomplete capture/replay must be UNKNOWN; malformed optional graph data must leave valid eager evidence PASS. Actual nested analysis bodies may be extracted for synthetic CPU trace controls; they are not GPU-equivalence evidence or a persistent test framework.

[Decision] Root accepted the collector CPU repair at source `0d745706`/records `6ef05dd8`; the original pending-review phrase above is historical. Root's original probe-image receipt `/data/ycfeng/tmp/glm53_dsv4/native_stream_probe_image_oj706s8g/receipt.json` already passed actual entry (expected GPU absence/exit2), actual config/wrapper construction and help on the249-line original. Do not rerun or rewrite it; final repaired-source/package bindings remain root-owned and pending.
