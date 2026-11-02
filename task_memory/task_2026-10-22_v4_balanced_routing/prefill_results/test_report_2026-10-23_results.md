# Balanced prefill bounded measurement results

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Validated collected H800 measurements, actual balanced identity, exact CSV regeneration and completed kernel trace; retained anomalies and limits. |

## Verdict

Task `W128-V4-BALANCED-PREFILL-RESULTS`: **PASS for isolated bounded measurement data, with retained anomalies**. Root acceptance remains separate. This is not serving, E2E, or target TP8/DP2/EP16 deployment admission.

Original root: `/data/ycfeng/tmp/glm53_dsv4/v4_balanced_prefill_39b2bf8_results/`.
Phase: `PREFILL_WARM_6144_C0_BALANCED/`.
All eight original artifacts remain unchanged. Total7,893,448 bytes; trace7,702,607 bytes.

Root-provided platform evidence reports `exp-1023-023708-916112` Succeeded on `gpu-h800-0570.host.platform.shaipower.com`, exit0, restarts0,500s=0.1388888888888889GPUh, within900s/0.25GPUh. This task read those receipts only; it performed no platform, GPU, Docker, SSH, cloud, auth, download or delegation operation. Remote/local hash comparison remains root-owned.

## Inputs and bindings

- Source: `39b2bf8088dff40dc8b695b6927d6e6322c337e9`.
- Sealed package: `/data/ycfeng/Frontier/.worktrees/v4-balanced-prefill-run/task_memory/task_2026-10-22_v4_h800_prefill/h800_prefill_balanced`.
- Manifest SHA256: `c6a009f87220c96d19f9b89097762b2f92e3ccdf05199398eaab331510331c8c`; all529 package file hashes rechecked.
- Native Git: `2cf0a6915ce544dc493a0990f2ea38d81601128a`; C26 SHA256 `d3556a47056bd52680b295f1585906149310397ec53ab90aa0008ff7d57e86ee`.
- Policy SHA256: `6562ec1c7e1ee7a8021d71a1a698c36f126307f00681320b07397b53c4c3f4ea`.
- Model revision: `0d992cbeef1f2d4bd92a8ae67a97a43df0c1bce5`; model config SHA256 `b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818`.
- Actual image reported by terminal receipt equals sealed job and root image command: `hub.i.basemind.com/vllm-openai/vllm-openai@sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635`.

Raw constructed-model inventory contains six `RoutingSimulatorRouter` instances and six `MoERunner` instances under exactly `model.layers.0..5.ffn.experts`. Environment/runtime path both equal `balanced_topk`; actual flat strategy origin equals the package policy path. All nine imported native routing module origins and SHA256 values match package expected pins. Actual `routing_columns(identity)` accepts the receipt. Both CSVs preserve this full identity exactly, along with source/model/runtime/device fields.

Root image receipt, whose saved hash was checked against readiness, reports60 source/519 tool/20 binary pins PASS. This task did not inspect installed native binaries itself. Runtime: Python3.12.3, Torch2.13.0+cu130, CUDA13.0, vLLM0.28.0, FlashInfer0.6.16.post3, Triton3.7.1; device NVIDIA H800/sm90. `deep_gemm:null` is retained; logs show vendored `vllm.third_party.deep_gemm` in use.

## Geometry, raw records and timing checks

| Check | Expected | Observed | Result |
| --- | --- | --- | --- |
| CSV row counts |6 per method |6 CUDA_EVENT;6 KERNEL_ONLY | PASS |
| Exact load |6144 prompt/C0, batch1 |All six steps and12 rows match | PASS |
| Bounds |7168 model/8192 budget/configured128/effective128 |Sealed plan engine128; actual LLM startup kwargs128; row max_num_seqs128 | PASS, evidence limit below |
| Physical engine |TP1/DP1/EP1/PP1/eager/six layers |Startup world_size1 and rank mapping; raw six layers; trace world_size1, graph IDs0 | PASS |
| Repeats |warm2+measure3 per method |Recorder window2/3, raw3 measured steps each; all applicable scope counts3 | PASS, evidence limit below |
| Raw records |One engine and exact applicable scopes |289 total:1 engine,6 steps,282 calls | PASS |
| CUDA_EVENT |3 steps with six layers |Steps0/1/2,150 calls | PASS |
| KERNEL_ONLY |3 steps with six layers |Steps3/4/5,132 calls | PASS |
| Raw timing |Finite and positive |All282 calls finite/positive | PASS |
| CSV statistics |Exact raw min/max/mean/median/std/count |94 applicable scope groups, three samples each, all exact | PASS |
| Native metadata |6144 scheduled,0 computed, no decode;7168/8192 |Every variant's per-step metadata matches | PASS |

The frozen recorder explicitly submits five sequential requests per method, drains each before the next, and excludes its first two matching steps. Raw request records contain scheduled6144/computed0/prompt6144, three measured steps per method. Trace measured annotations are only steps3/4/5. Skipped warmups and request/repeat IDs are not serialized separately; exclusion is verified from frozen recorder behavior plus measured coverage, not a separate skipped-request audit. CSV `max_num_seqs` is planner metadata, not an independently queried scheduler field; actual startup kwargs128 and sealed engine128 provide the effective-capacity evidence.

### Mean timings, milliseconds

Each value is the mean of all three retained samples for one layer. `plan/reuse` is the existing `plans_tile_schedule` identity. Dashes are inapplicable scopes, not zero measurements. Complete unrounded values, sample arrays, min/max/std and kernel counts are in `results_receipt.json`.

CUDA_EVENT:

| Layer/variant/role | Enclosing layer | Sparse attention | Input projections | RMSNorm | Q projection | Q/KV prepare | O projection | Compressor | Index prepare | Index scores/top-k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|0 uncompressed plan |1.543413 |0.580395 |0.267061 |0.107019 |0.152256 |0.203221 |0.178688 |— |— |— |
|1 uncompressed reuse |1.032992 |0.458219 |0.114165 |0.021803 |0.064651 |0.175296 |0.178549 |— |— |— |
|2 csa plan |3.350923 |0.917323 |0.318123 |0.022112 |0.129024 |0.296544 |0.179733 |0.195915 |0.461845 |1.421717 |
|3 hca plan |1.326507 |0.584555 |0.186827 |0.021131 |0.081099 |0.248416 |0.180640 |0.331755 |— |— |
|4 csa reuse |3.339691 |0.908928 |0.318741 |0.021397 |0.128139 |0.301483 |0.177877 |0.161440 |0.463008 |1.419008 |
|5 hca reuse |1.316203 |0.577685 |0.185003 |0.020448 |0.079861 |0.250272 |0.178432 |0.332587 |— |— |

KERNEL_ONLY:

| Layer/variant/role | Enclosing layer | Sparse attention | Input projections | RMSNorm | Q projection | Q/KV prepare | O projection | Compressor | Index prepare | Index scores/top-k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|0 uncompressed plan |— |0.440223 |0.110923 |0.016757 |0.060128 |0.169557 |0.173216 |— |— |— |
|1 uncompressed reuse |— |0.445972 |0.109951 |0.016598 |0.060256 |0.171253 |0.174582 |— |— |— |
|2 csa plan |— |0.896200 |0.300970 |0.016469 |0.065696 |0.197930 |0.181749 |0.098219 |0.225620 |1.385095 |
|3 hca plan |— |0.570100 |0.178293 |0.016427 |0.062987 |0.170293 |0.172907 |0.145439 |— |— |
|4 csa reuse |— |0.894867 |0.302591 |0.016437 |0.068331 |0.190815 |0.174656 |0.088192 |0.231743 |1.380263 |
|5 hca reuse |— |0.570932 |0.180448 |0.016341 |0.063657 |0.170080 |0.174528 |0.147392 |— |— |

Scopes may overlap across auxiliary streams; scope sums are not enclosing layer wall times. Methods are separate runs, not paired measurements of the same execution. No CPU-overhead/gap attribution was performed.

## Exact CSV regeneration

The existing sealed aggregate CLI ran unchanged on a copied raw JSONL in a new result-owned output. Originals were never passed as a writable raw directory: the CLI writes its aggregate summary under raw_dir.

Initial Python3.10.6 execution returned0 but differed in14 `.std` cells at the final floating-point digit. All other cells and column order matched. The sealed implementation uses `statistics.pstdev`. Existing local Python3.12.13 recomputed all94 std strings exactly; a second unchanged CLI execution under that interpreter with the package's pinned pandas2.3.3 wheel produced exact bytes for both files. No post-processing or normalization was used. `aggregate_iteration_receipt.json` preserves the first difference and the pre-retry hypothesis.

Successful command:

```bash
env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp PYTHONPATH=/data/ycfeng/Frontier/.worktrees/v4-balanced-prefill-run/task_memory/task_2026-10-22_v4_h800_prefill/h800_prefill_balanced/source/39b2bf8088dff40dc8b695b6927d6e6322c337e9:/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-22_v4_balanced_routing/prefill_results/aggregate_rebuild_py312/wheel_site /data/ycfeng/envs/asyncopd-py312/bin/python -m frontier.profiling.attention.dsv4_attention_main aggregate --model_name deepseek-ai/DeepSeek-V4-Flash --model_config_path /data/ycfeng/Frontier/.worktrees/v4-balanced-prefill-run/task_memory/task_2026-10-22_v4_h800_prefill/h800_prefill_balanced/model/config.json --target_tensor_parallel_size 8 --num_layers 6 --block_size 256 --kv_cache_dtype fp8 --max_context_len 7168 --max_kv_tokens_per_lane 7168 --max_num_seqs 128 --chunk_budgets 8192 --prefill_prompt_len 6144 --prefill_warmup_repeats 2 --prefill_measured_repeats 3 --raw_dir /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-22_v4_balanced_routing/prefill_results/aggregate_rebuild_py312/raw --output_root /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-22_v4_balanced_routing/prefill_results/aggregate_rebuild_py312/out --device h800 --source_revision 39b2bf8088dff40dc8b695b6927d6e6322c337e9 --model_config_revision 0d992cbeef1f2d4bd92a8ae67a97a43df0c1bce5
```

cwd was the sealed source snapshot above. Local Python3.12.13/Clang22.1.3, pandas2.3.3 from retained cp312 wheel, NumPy2.2.6; no conda activation or installation. Wheel bytes matched manifest; unpacking reused the existing worker's wheelhouse mechanism. Exact argv/environment/versions, exit0 and stdout/stderr paths are in `aggregate_rebuild_receipt.json`. CPU validation was inline Python, not a new maintained harness; it invoked existing `routing_columns`, `expected_scopes`, `_case_window`, `aggregate_kernel_trace`, plan parser and builder, and recomputed raw scope statistics.

| CSV | Bytes | Original and rebuilt SHA256 | Result |
| --- | ---: | --- | --- |
| attention.csv |43114 |`52571225ce8ddd0d2e87615f80f2aaadd980cbd8b2c10d460a1c75dff7c2af5f` |BYTE-IDENTICAL |
| attention_kernel_only.csv |41635 |`fe22f22cf4c21bcad93c8ccd463115a607ace65193d900925b8eaa2e6cf69b43` |BYTE-IDENTICAL |

A receipt assertion initially compared JSON lists in `reduced_layers` with Python tuples in `plan.summary()`. Direct inspection isolated that representation-only mismatch. JSON-roundtrip comparison passed in final validation; no source/input behavior changed. Both verification iterations are recorded, not hidden.

## Completed CUDA/kernel evidence

Trace JSON parsed successfully:26,394 events,1,815 positive-duration kernel events,132 measured CPU user annotations, CUDA runtime2703/driver710 events. Trace host/device/world_size match H800 platform/raw evidence; every kernel has device0 and graph id0. Log contains profiler_start, profiler_stop and three measured KERNEL_ONLY steps.

Existing `aggregate_kernel_trace(events)` reconstructed132 scoped timings and counts, exactly matching every KERNEL_ONLY raw call. They account for366 scoped kernel occurrences; the aggregate33.915244ms is a sum across scopes/repeats, not layer wall time. No measured annotation lacks correlated CUDA kernel evidence. Full trace also includes unannotated warmup/other kernels, which are not consumed as measured scopes. There is no separate CUDA_EVENT Chrome trace; that method is evidenced by positive raw CUDA-event calls and frozen `elapsed_time` recorder semantics.

Trace SHA256: `990dd2a7a4a4dd9adcd427dd047cb5bb34b65b791563bca7d0d76a3f4f54cdff`. Status reports archived7,702,607 trace bytes and zero trace bytes left on worker; local trace size equals archived value.

## Retained anomalies and limitations

- CSA layer2 CUDA_EVENT compressor samples are0.1602240056/0.2651520073/0.1623679996ms, mean0.1959146708ms, max/min1.654883. All retained. Kernel-only CSA compressor variability is also retained: layer2 max/min1.116308, layer4 max/min1.125331.
- Uncompressed layer0 event projection/rmsnorm/q-projection means exceed layer1 and kernel-only counterparts; its projection max/min1.145492. Do not treat this as a paired method difference or identify a CPU/kernel cause without further evidence.
- Chat-template warmup emitted a warning traceback because `skip_tokenizer_init=True`; token-ID profiling completed. Other retained warnings cover experimental Frontier architecture limits, TileLang serial vectorization, invalid model outputs under simulated routing, and missing explicit NCCL shutdown on clean exit.
- Engine initialization took411.45s with substantial JIT/warmup. This is not request latency or an E2E measurement.
- The prior startup-cap1 failure did not recur: actual startup max_num_seqs128, engine initialized, both methods completed, no CUDA invalid-configuration error. This is not proof about the exact old failing CUDA kernel/root cause.
- Per-expert routing assignment histograms are not serialized. Actual balanced strategy/router/native binding is verified; actual assignment counts were not separately captured.
- Physical TP1 and reduced six-layer dummy model measurements do not admit target TP8/DP2/EP16 serving or full-model E2E equivalence. Root retains final acceptance and remote hash comparison.

No original artifact was edited, filtered or normalized. No source, tests, builder, package, Git state, other task record or broad test suite was changed. Native B checker repair remains independent.

## Progress checkpoint and handoff

Recorded data-validation subtasks are complete: source/input binding, raw/CSV geometry and statistics, real constructed routing identity, exact aggregate rebuild, trace evidence, and anomaly report. D1–D7 are CLEAR; the two verification corrections narrowed specific observed causes rather than changing data. No delegated-scope S1–S6 defect requires a spec change. Only root acceptance/remote comparison remains outside this slice. Final integrity receipt confirms originals and sealed inputs unchanged; writing stopped after its creation.
