# Retained V4 kernel coverage and dependency audit

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Audited accepted retained trace and pinned native/C26 sources; no GPU or original edits. |

Task ID: `W128-V4-KERNEL-COVERAGE`. Status: **audit complete; serving totals blocked**.
This is a retained-data prerequisite audit, not a formal matched `op-gap-analysis` entry.
No new matched runs, serving admission, CPU-cost classification or core changes.

## Inputs and integrity

Authority: `task_memory/task_2026-10-22_v4_balanced_routing/prefill_results/primary_acceptance.json`.
Accepted source `39b2bf8088dff40dc8b695b6927d6e6322c337e9`; native declared
`2cf0a6915ce544dc493a0990f2ea38d81601128a`; C26 model leaf comes from sealed `overlay_model.py`, not pristine `nvidia/model.py`.

`kernel_coverage_receipt_r2.json` records SHA256/size before and SHA256 after for **562 files**, including
529 sealed manifest files, all seven primary retained files, all eight original scratch artifacts,
trace, source tools and inspected native leaves. Initial554-file receipt is retained unchanged.
All match; **PASS originals unchanged**. This checks local bytes against recorded pins, not live installed image or Git state.
Extra native source hashes are observed before/after and linked to the declared checkout; files absent from the sealed image pin set are not independently installed-image-qualified.

Trace: `/data/ycfeng/tmp/glm53_dsv4/v4_balanced_prefill_39b2bf8_results/PREFILL_WARM_6144_C0_BALANCED/profiler_traces/e0-prefill-p6144.json`;
7,702,607 bytes, SHA256 `990dd2a7a4a4dd9adcd427dd047cb5bb34b65b791563bca7d0d76a3f4f54cdff`.
Actual header: H800/SM90/rank0/world1/graph IDs0. Raw: TP1/DP1/EP1/eager/six layers,
query6144/context0, three measured repeats per method, two unmeasured warmups.

## Census: observed denominators only

| Item | Count | Interpretation |
| --- | ---: | --- |
| Trace events / kernels | 26,394 / 1,815 | Entire retained profiler session, not measured attention denominator |
| Measured annotations / unique scoped kernels | 132 / 366 | Exact KERNEL_ONLY raw replay; no empty annotation or duplicate kernel owner |
| Kernel missing/ambiguous launch correlation | 0 / 0 | Every kernel has exactly one retained runtime/driver launch |
| Kernels without External-id CPU-op parent | 525 | 132 are scoped via launch/thread containment;393 remain unowned |
| Kernels skipped by component aggregate | 1,449 | Full explicit evidence in receipt; not all warmup/non-attention |
| Unowned before first / between first-last / after last annotation | 815 / 600 / 34 | Host-launch regions only; do not relabel as exact warmup/measurement windows |
| Unowned within each of18 layer component envelopes | 0 | Supports component-envelope coverage, not absolute outer-forward completeness |

All1,449 unowned kernels, all525 no-External-parent indices, and empty ambiguity list are retained in
`kernel_coverage_receipt_r2.json`. Names are verbatim, not guessed family assignments. Generic `Kernel`
appears32 times. Before-first evidence contains5 priming kernels (one fill/four adds) and two full
source-described warmup requests, but lacks independent request/forward boundaries;815 is **not** a warmup denominator.
Exact request IDs and warmup exclusions are recorder/source evidence, not separately serialized measurements.

No measured scope is dropped. The source-derived attention body consists of wrapped compute plus
allocation/split/view operations (`attention.py:361-403`); every retained launch inside first-projection
through last-output-projection envelopes is owned. However, the intentionally unannotated outer
`DeepseekV4Attention.forward` (`dsv4_attention_runner.py:155-164`) leaves completeness outside those
envelopes unproven. No strict all-attention scope claim can be made from this trace.

### Attention versus FFN/mHC and metadata

Hooks cover input projections, q/kv norm, q projection, fused query/cache preparation, compressor,
index preparation/scores/top-k, sparse attention and output projection. Parent forward allocation/views
do not have separate annotations; sparse gather/combine kernels remain inside the sparse hook.

Unhooked native calls: decoder mHC attention preparation and attention-to-FFN transition
(`overlay_model.py:909-988`), FFN (`:990`), embedding (`:1141`), final mHC post (`:1189`),
hc_head (`:1203`), final norm (`:1211`) and logits. Conditional aux mHC reconstruction is not asserted active.
These are not `DeepseekV4Attention` components; identical mHC kernel names cannot establish layer/side ownership.
Retained Marlin MoE CPU-op parents and activation/GEMM records confirm the session also includes FFN work.

Attention-related **GPU metadata preparation** is separately outside forward hooks:
compressed slot mapping (`compressor_utils.py:65-94`), SWA indices/lens (`sparse_swa.py:584-625`),
C128 top-k metadata (`sparse_mla.py:331-347`). Actual unowned fingerprints include those kernels
(15/15/5 occurrences), but have no layer/repeat outer parent. Do not admit them as missing per-layer
forward attention cost without first deciding/recording the boundary.

## Actual stream/dependency evidence

Raw input6144 exceeds multi-stream GEMM threshold1024; **all60 measured input-projection kernels use
stream7**. The decode/small-input aux-GEMM branch is not active here (`attention.py:565-573`).
CSA measured kernels use7/21/25/29; HCA7/21; uncompressed7. The trace has125 event records,
85 stream waits,5 event synchronizations and3 device synchronizations. Measured CSA layer envelopes
each contain6 waits; HCA2; uncompressed0 (48 measured-envelope waits total).

Source proves fan-out/join rules: `execute_in_parallel` (`multi_stream_utils.py:117-130`) and
`maybe_execute_in_parallel` (`:51-58`). CSA q/cache and indexer/compressor branches join before
index scores/attention (`attention.py:466-506`); index compressor joins index query before indexer_op
(`:950-958`). These are **source semantics**. Actual sync trace args are only `cbid,correlation`:
no event handle, record generation, or source/destination stream handles. `ac2g` links launch to activity,
not inter-stream event dependency. No actual cross-stream event-edge graph was reconstructed or invented.

Each stream's host enqueue order matches kernel start order (zero reversals). **Do not turn enqueue
order into full-completion FIFO edges**: actual trace contains93 adjacent same-stream interval overlaps,
10 between measured scoped kernels. PDL is enabled in accepted logline23; norm uses GDC primitives
(`fused_qk_rmsnorm.py:46-48,86-101`), inverse-RoPE uses GDC (`fused_inv_rope_fp8_quant.py:60-62,259-283`),
DeepGEMM enables PDL (`deep_gemm.py:271-273`). Compressor deliberately disables PDL
(`compressor.py:368-385`). Sub-microsecond overlaps may also reflect trace precision; no attribution of
every overlap to PDL is claimed. Source producer/consumer tensors and PDL instructions are not a measured
kernel-level visibility/release timeline.

## Per-layer/per-repeat diagnostic comparisons (ms)

Sum = owned kernel durations; union/span = intervals of the **same owned kernels**. Not serving cost.
All18 samples remain; repeat indices are measured steps3/4/5, not identical requests shared with CUDA_EVENT.

| Step | Layer/type | Sum | Union | Span |
| --- | --- | ---: | ---: | ---: |
| 3 | 0/uncompressed | .977821 | .977821 | 1.835579 |
| 3 | 1/uncompressed | .982141 | .982141 | .992669 |
| 3 | 2/CSA | 3.380695 | 3.253943 | 3.290071 |
| 3 | 3/HCA | 1.318269 | 1.276829 | 1.296381 |
| 3 | 4/CSA | 3.367256 | 3.244343 | 3.270839 |
| 3 | 5/HCA | 1.320284 | 1.278813 | 1.290589 |
| 4 | 0/uncompressed | .970334 | .970334 | 1.612860 |
| 4 | 1/uncompressed | .979326 | .979326 | .989853 |
| 4 | 2/CSA | 3.375991 | 3.242168 | 3.276248 |
| 4 | 3/HCA | 1.316510 | 1.276093 | 1.296028 |
| 4 | 4/CSA | 3.341910 | 3.232280 | 3.259224 |
| 4 | 5/HCA | 1.311485 | 1.272637 | 1.285949 |
| 5 | 0/uncompressed | .964254 | .964254 | 1.580764 |
| 5 | 1/uncompressed | .974367 | .974367 | .984446 |
| 5 | 2/CSA | 3.347159 | 3.230135 | 3.265208 |
| 5 | 3/HCA | 1.314557 | 1.274493 | 1.293533 |
| 5 | 4/CSA | 3.334519 | 3.227767 | 3.256856 |
| 5 | 5/HCA | 1.338366 | 1.295806 | 1.307773 |

The366-duration sum33.915244ms reproduces raw evidence; it is not an additive serving total.
Sum exceeds union for compressed layers, directly refuting disjoint-scopes-implies-no-overlap.
Layer0 has much larger span than busy union in this KERNEL_ONLY session too; no CPU attribution follows.

**Historical proposal arithmetic correction:** `proposal.md:23-25` CUDA_EVENT "component sum" includes
`dsv4_attention_layer` itself. Correct sums of **component medians excluding outer** are
3.887168/3.874048 (CSA layers2/4),1.635584/1.620832 (HCA3/5),1.484800/1.014272 (uncompressed0/1).
Independent composite medians stay3.327680/3.312032,1.324704/1.310880,1.524128/1.034080.
Old7.214848/2.960288 etc therefore cannot support an overlap comparison. Parent owns proposal correction.
Receipt also retains every independent CUDA_EVENT per-repeat component sum/composite; no paired subtraction
or CUDA_EVENT-minus-KERNEL_ONLY CPU-cost label is used.

## Exact minimum capture addition; no run authorization

1. Nested **outer forward/layer/attention ownership ranges** carrying request/repeat/phase/warmup identity;
   keep outer ranges out of disjoint component timing aggregate. Current parser rejects nested component ranges.
2. Separate metadata-preparation, mHC, FFN, embedding/norm/logits family ownership annotations, so all unowned
   activities acquire actual native parents rather than kernel-name guesses. Explicitly distinguish metadata
   boundary from `DeepseekV4Attention.forward` boundary.
3. At existing event record/wait sites capture **event handle + record generation + source/destination stream
   handles + step/layer/scope**, with mapping to trace stream IDs. Retain actual PDL launch attributes and source
   primitive mapping; overlapping producers/consumers need visibility/release evidence, not injected completion edges.
4. Keep every correlation/unowned activity; require unique outer family/phase parent for strict completeness.
   The actual TP8/DP2/EP16/V2/graph witness remains separate; no GPU run was launched here.

## Verification and limits

Python `/data/ycfeng/envs/frontier-py310/bin/python`,3.10.6; standard library only, no torch import,
installation, platform/SSH/cloud/Docker/Git/native/C26 edits. `--help` and docstrings inspected before tool use.
Existing sealed `dsa_kernel_trace` parser/index/aggregate reused; existing forward_device_timeline docstring/help
inspected, but that CLI was not used for reduction because this trace has no `Forward` range and different format.

Executed from `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`:

```sh
timeout 60s /data/ycfeng/envs/frontier-py310/bin/python -B tests/analysis/v4_kernel_coverage.py --acceptance task_memory/task_2026-10-22_v4_balanced_routing/prefill_results/primary_acceptance.json --package /data/ycfeng/Frontier/.worktrees/v4-balanced-prefill-run/task_memory/task_2026-10-22_v4_h800_prefill/h800_prefill_balanced --native /data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 --trace /data/ycfeng/tmp/glm53_dsv4/v4_balanced_prefill_39b2bf8_results/PREFILL_WARM_6144_C0_BALANCED/profiler_traces/e0-prefill-p6144.json --output task_memory/task_2026-10-23_v4_remaining_measurements/audit/kernel_coverage_receipt_r2.json
```

Expected132 annotations/366 scoped kernels/1815 total, exact raw replay and all original SHA256 unchanged;
actual values match, exit0: **PASS census/integrity**. Coverage and actual dependency completeness:
**UNKNOWN; serving total/admission blocked**. Tool227lines, one owned analysis tool; no test-suite/core edit.
`native_source_evidence.json` provides source line spans and minimal capture requirements, hashes in main receipt.
No memory-heavy data load (retained trace7.7MB);60s timeout applied.
