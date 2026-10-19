# Step-5-Preview development status

Step-5-Preview has complete configuration and layer representation, plus a controlled structural request path in Frontier. Native single-role qualification, measured prediction, PD integration and formal hardware validation remain incomplete. This branch must be described as development work until the acceptance gates below pass.

The shared implementation serves the GLM-5.3, DeepSeek-V4-Flash and Step-5-Preview integration plan. Model profiles extend the existing configuration, attention, operator, execution-time and cache interfaces. Shared infrastructure has one canonical implementation; model-specific contracts remain at their existing binding boundaries. The final integration target remains `feat/glm53-dsv4-vllm028`, with one reviewed source revision for three-model regression.

## Scope and reference

The target is full text-backbone online simulation on 64 H800 GPUs with sequential prefill/decode disaggregation, PP1, chunked prefill and native decode CUDA Graph replay. The primary PD topology is P32/D32 with TP8/DP4/EP32. Actual H200 qualification and comparisons are also required before overall completion. Current testing uses H800 first; H200 is a quota fallback. Tests use at most64 GPUs and explicitly non-preemptible jobs.

The Step native reference is the official `vllm/vllm-openai:step-5-preview` image pinned to linux/amd64 manifest `sha256:69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20`. The observed installed package is `0.1.dev22912+g8224bb82f`, with retained file identities; its full build revision is unknown. This reference is separate from the GLM/DeepSeek vLLM 0.28.0 reference. Frontier itself remains CPU-only and does not import native vLLM or Torch to simulate requests.

Use the locked official model configuration and tokenizer with native dummy weights; model checkpoint downloads are excluded. Dummy weights preserve the full model structure, expert count, quantization exclusions, actual kernels, communication and allocations. Vision/video/multimodal execution, MTP, prefix caching, offload and PD-AF are outside this Step scope. Any wrapper-associated native residency must still be accounted for truthfully.

## Implemented representation

[Model configuration](../../data/config/models/stepfun-ai__Step-5-Preview.json) records 92 backbone layers, 69 SWA and 23 sparse-GQA layers, 88 MoE layers and 4 dense layers. Configured geometry includes hidden size 4096, 64 query heads, 4 KV groups and explicit head dimension 192. Routed/shared FFN width is 1536, with 352 experts and top-8 routing; dense width is 13824. Sparse selection uses 256-wide proxies, 16 index heads, 4 provider groups and top-512 regions of 8 tokens. These are source-derived contracts, not native measurements.

[Configuration resolution](../../frontier/config/model_config.py) normalizes the nested text configuration while preserving the outer FP8 policy and exclusions. [Attention bindings](../../frontier/attention/model_binding.py), [Step attention contracts](../../frontier/attention/step5.py), [cache description](../../frontier/attention/step5_memory.py) and [parameter accounting](../../frontier/utils/param_counter.py) represent per-layer precision, raw KV aliases, sparse summaries and rank-local storage. The controlled MONOLITHIC/PP1 path uses the existing group tables and block pool. Resident, unique and wire bytes retain separate meanings.

[The structural driver](../../tests/integration/run_step5_structural.py) connects those contracts to scheduling, model-layer execution, communication and cache lifetime. The accepted source-structural cases each complete 17 stages, 1564 backbone executions, 1496 FFN merges, 4105 prefill tokens, 8 decode inputs, 9 outputs and 35 cache-lifetime events. This uses synthetic contract costs and establishes complete structural accounting only. It does not establish trained prediction, measured latency, native parity or PD support.

## Evidence and current progress

| Area | Status | Verified result | Remaining gate |
| --- | --- | --- | --- |
| Baseline reuse, immutable reference, model inventory; tickets 01–03 |DONE |Recorded source/reference audits and complete inventory |Preserve identities when integrating later changes |
| Rank-local kernels/state; ticket 04 |DONE within G1 scope |Accepted rank-local qualification |G1 does not establish a complete role or PD |
| Early physical scopes; ticket 05 |PARTIAL |Local measurements and coverage dispositions retained |Admit exclusive measurements through the complete native path |
| Model representation; ticket 15 milestone |DONE within contract scope |738 checks, 77 legacy scenarios and 451 artifact pairs in the accepted representation generation |Official comparator/cache and recorded performance limitations remain |
| Controlled structural path; ticket 15 milestone |DONE within source-structural scope |646 affected checks, 3 actual CLI cases, 77 legacy scenarios and 451 artifact pairs |Native reconciliation and all 8 whole-ticket criteria remain open |
| Native single-role; ticket 06 |PARTIAL, 1/10 |Prior fixed-length public completion only |Actual precision/content/cache/physical traces, complete G2 and tooling fidelity |
| Observer validation and native PD; tickets 07–08 |BLOCKED on complete 06; preparation available |Existing observation and source/metric preparation |Correlation/perturbation, complete handoff, CUDA-visible content and release ordering |
| Predictors, H800 64-GPU, H200 and three-model acceptance |NOT VALIDATED |No formal predictive acceptance claimed |Independent profiles/calibration/holdouts, per-cell gates and common-source regression |

The latest native correction is commit `94d154e`: a fresh CPU process drains Kineto FIFO output while the original exporter remains on its calling thread. Cleanup preserves the primary native error and verifies reader termination before releasing its pending allowance. Legitimate `preempted_req_ids=None` is retained. Five files extend the existing native observer and tests; no Frontier core, graph policy, workload or native model implementation changes.

Accepted affected coverage is 626 distinct CPU cases: 425 observer and 201 driver cases. The final 168 profiler rerun covers the final test-fixture change and overlaps those cases. A separate locked-image CPU exporter check retains complete traces of 2,664,717 and 2,675,714 bytes through an actual 65,536-byte pipe with CUDA uninitialized. A 31-byte negative case remains unverified. Final independent Standards and Spec reviews have no remaining findings. This establishes the source/CPU correction; it does not prove every cause of the previous GPU timeout.

On October 20, 2026 at 00:09:28 HKT, a distinct 16-GPU H800 diagnostic, `request_physical_20261019_04`, was submitted at source `94d154e`. The 00:11:49 HKT snapshot reports both replicas Running with 16 distinct GPU UUIDs. Its 201 inputs / 16,439,703 bytes were frozen before submission; 122 native pins, 24 inert SDK checks and 543 source/generation checks pass. The original 2010-second workload, 2400-second worker, 300-second archive reserve, 8 MiB trace cap, 512 MiB node records and 4 GiB archive per node are unchanged. This is a time-stamped in-progress observation. Native completion, physical evidence and terminal status are pending.

Prior failed generations, zero-byte traces, original errors and over-budget or unverified checks remain on record. A fixture, successful import, parsed configuration, clean exit or source-defined cost is never independent native ground truth. Qualification scripts under [step5_environment](../../tests/integration/step5_environment) require separately prepared locked task artifacts; they are not a self-contained public benchmark recipe. Detailed campaign records and large traces remain local or in the approved artifact store, outside this repository.

## Resume order and ownership

1. Complete ticket 06 from the current native generation using the existing all-rank, request/device, state, profiler and archive validators. Audit actual native outcomes and all ten criteria; preserve failure evidence and make only the smallest supported correction.
2. Reconcile ticket 15 against complete 06. In parallel after 06, validate ticket 07 observer correlation/overhead and ticket 08 single-request PD content/lifetime. Independent source preparation may continue earlier.
3. Establish the complete PD path before concurrency, reuse and cancellation. Then qualify the required 64-H800 online configuration and adapt the existing metric normalizer and gate without changing metric meanings.
4. Collect complete-path exclusive operators, CPU scheduling/launch costs, routing, collectives and NIXL evidence. Compare workflow and operators together, explaining E2E errors before calibration.
5. Validate measured predictions with independent calibration and holdout evidence, required H200 work and same-source legacy/three-model regression. Integration tickets 16–22 retain the GLM full-path prerequisite and their native/shared-semantic prerequisites.

The native observer/caller, shared cache/core interfaces and existing metric normalizer/gate each have one owner. Do not introduce a parallel registry, duplicate collector, model-specific runtime copy or silent fallback. Routine extensions, registrations, instrumentation and evidence collection proceed within scope. A necessary material core refactor requires a demonstrated limitation, the smallest proposed change, alternatives and explicit approval before implementation.

Formal primary acceptance requires every specified per-cell mean metric error at most 10%, with conditional p95 client TTFT/ITL error at most 15% under the approved sample/window rules. Clean measurement campaigns decide predictive acceptance; instrumented diagnostic timing and calibration data cannot serve as independent holdouts. No formal gate is closed by the current development status.
