# D29 actual native routing and balanced-policy decision

## Modification History

| Date | Change |
| --- | --- |
|2026-10-22|Resolved actual FP8/BF16 modular paths and documented the G3/wave-balance semantic conflict; proposal only.|

## Outcome and required decision

No existing locked-native setting provides the requested exact balanced assignment. Both actual Step precisions support precomputed top-k without changing the expert backend. However, **a forward/wave-reset Hamilton policy cannot preserve the independent G3 full-local reference across P6128+D16, prefix chunks and q1 continuation**. This is a semantic incompatibility, not a missing flag or numerical tolerance.

Recommended question to YC, before any native/image/source change:

> 是否将“balanced”定义为按请求逻辑 token position 的确定性循环路由，并让 Frontier 按这些真实 position 区间计算专家计数，而不再要求每个 EP wave 都从专家0执行 Hamilton 均分？这样保留 P/D 拆分、chunk 和 full-local G3 reference 的路由一致性，但两个相同 position 的请求或短 q1 wave 不一定瞬时均分。否则，是否坚持逐 wave 精确均分，并明确将其限定为独立 timing-only 工作流、保留另一个未改变 reference 的 G3 工作流？

Both outcomes are **pending explicit user decision**. The second outcome changes campaign/qualification scope and is not authorized here. Do not implement either, copy actual top-k into the reference, relax G3, introduce tolerance, flip backend, or enable random routing as a substitute.

## Frozen inputs and evidence tiers

- Owned repo: `/data/ycfeng/Frontier/.worktrees/step5-d24-stage`; D28 code commit `6e73330c2d04af382cf65a2c689c5401ca202485`, records commit `65878aa273f6eb1f37a7c1547ed0f2c951dc51fa`.
- Sealed, unsubmitted, **not balanced**: `/data/ycfeng/tmp/step5_d24_native_20261022/capture03/stage`; manifest `cdaad24ecded6da6f2eea90ed4ef4a582432db2fd75c56c29aafb22f2f56a57d`, config `046b6f25c1f993790c135e6740cdf7bed247078f3b99f2713be2bfa97277854f`.
- Locked native root `N=/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm`; no file changed. Image digest `a.z01.normandyfunc.com/docker-public/vllm/vllm-openai@sha256:69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20`. Exact inspected source hashes in `cpu_receipt.json`.
- Frozen role geometry: H80032=P16+D16, TP8/DP2/EP16/PP1, E352/K8, no sequence parallel/mega override, `moe_backend=auto`, `allgather_reducescatter`, expert placement linear, no EPLB, max_num_seqs128, original P automatic pool and D32000. Runtime/90min/48GPUh/pressure criteria remain unchanged.
- Historical native module evidence `R=/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/step5_comm_20261020/s2_20261020_01`: `manifest.json` has the same TP8/DP2/EP16, auto backend, official image; all16 `native_dispatch/rank_*.json` agree on layers3 and88. This is measured module dispatch, **not current full-serving/graph attestation**.
- Retained capture01 `native_startup01.json:19–21,63–66,88–89` observes P6128/NONE, Dq1/FULL and native `_count_expert_num_tokens`/`fused_moe_kernel` JIT. It does not identify every routed layer's selected expert subkernel. Earlier profiling design assumed Triton only; later real S2 module receipts take precedence.
- Read-only CPU probes execute frozen factory/runner/DP/dummy method bodies with CPU seam objects; hypothetical tensor IDs validate policy mathematics only. No strategy installed, no engine/backend construction, no GPU/collective/API call. Live numerical/timing claims remain UNVERIFIED.

## 1. Actual Step FP8/BF16 route

All paths below are relative to N unless marked Frontier/R.

| Stage | Decisive path/lines | Frozen reachable behavior |
| --- | --- | --- |
|Text config|`model_executor/models/step3p7.py:70–75`; `models/step5/nvidia/vl.py:8–17`|VLM frontend constructs registered text model with nested text_config; Step5 model state is used.|
|Model geometry|`models/step5/nvidia/model.py:1680–1807` and capture03 reference config|MoE layers3–90, FP8 dynamic block128 routed3–87, BF16 routed88–90;352/top8, expert hidden1536, hidden4096, swiglustep clamp7; gate FP32, biased sigmoid choice, unbiased normalized sigmoid weights scaled3.|
|Factory vs mega|`model.py:755–762,1756–1807`|Only explicit `deep_gemm_mega_moe` enables that model branch/SP. Frozen auto uses FusedMoEFactory; no backend flip needed.|
|Router creation|`model_executor/layers/fused_moe/layer.py:294–315`; `router/router_factory.py:120–126`|Factory creates ordinary biased-sigmoid router when simulation env is empty; nonempty env first selects RoutingSimulatorRouter.|
|FP8 oracle|`oracle/fp8.py:77–130,146–169`; `experts/flashinfer_cutlass_moe.py:184–191`; `experts/trtllm_fp8_moe.py:165–172`|Hopper EP initially prefers FI CUTLASS, but swiglustep is unsupported. TRTLLM is SM100-family-only, thus unavailable on H800. Auto reaches modular `TritonOrDeepGemmExperts`; R all16 ranks observe Fp8MoEMethod/DEEPGEMM and that class.|
|BF16 oracle|`oracle/unquantized.py:50–86,134–139`|Hopper demotes both FI implementations; modular Triton comes first. R all16 ranks observe UnquantizedFusedMoEMethod/TRITON/TritonExperts.|
|Runtime routing|`runner/moe_runner.py:620–642`; `routed_experts.py:1338–1373`; `quantization/fp8.py:824–847`|Both frozen methods are non-monolithic: `router.select_experts` returns weights/IDs, then `forward_modular`/quant apply consumes them. Actual CPU method-seam execution confirms order for both precisions.|
|Dispatch order|`prepare_finalize/naive_dp_ep.py:136–208`; `distributed/device_communicators/all2all.py:100–124`|Routing happens on each DP lane before DP all_gatherv of hidden states, weights, IDs and scales. TP8 copies must agree; do not multiply the token count by8. R rank0 observes sizes[8,1], gathered9 rows, group[0,8], no SP.|

**FP8 expert subkernel caveat is now exact:** `experts/triton_deep_gemm_moe.py:77–86` chooses DeepGEMM when `is_deep_gemm_e8m0_used()` **or** `_valid_deep_gemm(...)`; otherwise its existing Triton implementation. `_valid_deep_gemm` requires M>=native alignment and valid N/K (`experts/deep_gemm_moe.py:68–104`). **Small M does not alone imply Triton**: `utils/deep_gemm.py:120–141` can enable E8M0 on Hopper when the imported symbol exists; env default is1 (`envs.py:1619–1620`). R observes DeepGemmExperts even at gatheredM9. Therefore preserve/attest this actual condition; do not hard-code a “large=DeepGEMM/small=Triton” identity. Both alternatives remain modular and use the same upstream selected IDs.

The generic monolithic bypass exists (`MoERunner:620–625`, `routed_experts.py:1375–1405`) but is **not the frozen H800 Step route**. It accepts logits, not a precomputed top-k pair. Extending that unrelated backend would need a kernel/router hook and is unnecessary/not proposed here. Actual installed/kernel/env choices on the next worker still require primary attestation; source-only selection is not a new GPU observation.

## 2. Existing settings and what is missing

- `envs.py:1852–1854` exposes one simulation selector; `routing_simulator_router.py:226–235,279–298` registers only `uniform_random` and `normal_routing`. `balanced` is unknown and raises ValueError. `uniform_random` samples per-token random top-k without repeats, not exact expert counts. It returns all-one weights rather than Step scaling/renormalization and warns outputs are invalid.
- Read-only searches of `envs.py`, config, fused_moe and Step source for `load_balance`, `balanced`, `round_robin`, `force_route`, `forced_rout`, routing simulation found no alternate native token-balanced flag. `ParallelConfig.expert_placement_strategy`/`needs_round_robin_routing_tables` map ownership for expert backends, not balanced token choice (`config/parallel.py:202–214`; `fused_moe/config.py:1060–1062`). Other load_balance hits are vision/attention work scheduling. `enable_eplb` changes placement/replicas, not this deterministic per-token policy.
- Existing factory can accept custom/preconfigured routers, but no serving flag constructs an exact balanced one. An out-of-tree hook/new registry entry is **new native behavior**, not an existing setting. Do not hide it in diagnostic sitecustomize: routing failure must not soft-fallback to ordinary/random routing while marking the result balanced.
- Existing simulation factory drops scoring_func/renormalize/scaling (`router_factory.py:120–126`); strategy API receives hidden states/logits/top-k only. A one-line registered strategy with default weights would silently change Step semantics. The smallest complete extension must explicitly carry Step weight settings or have a supported router that retains them.

## 3. Aggregate scope: DP counts solve one problem, not G3

Frontier `frontier/moe_ep_workload.py:774–807,810–855` materializes exact Hamilton counts over one Replica-local physical EP wave, then splits ownership. For equal ratios, first remainder experts ascending receive+1. A hypothetical contiguous stream `ID(t,j)=(t*K+j)%E` matches these counts exactly only when each wave starts at stream0.

At native DP2/no-SP, `ForwardContext.DPMetadata` has CPU `num_tokens_across_dp_cpu`, `sp_local_sizes(1)` and cumulative counts; DP rank's prefix can be computed before routing without a new collective/readback. Actual MRv2 synchronization is **`v1/worker/gpu/dp_utils.py`**, not the older V1 utility: lines66–79 use the existing CPU all_reduce; eager returns original counts (155–171), graph returns agreed padded counts (197–203). FULL capture uses fixed equal shape per DP (`gpu/cudagraph_utils.py:677–680`). Counts contain no request logical position or prompt-prefix offset.

CPU actual-method/pure-contract results (`cpu_receipt.json`):

|DP physical counts|Global-prefix Hamilton match|Per-DP reset mismatched experts|
| --- | --- | --- |
|[6128,1]|Yes|16|
|[6144,1]|Yes|16|
|[8,8]|Yes|128|
|[1,1]|Yes|16|
|[4,4]|Yes|64|
|[256,256]|Yes|128|

Top-k distinctness, sum=N*K and max-min<=1 pass for global-prefix hypothetical IDs. These checks demonstrate the minimum aggregate scope; **they do not make a wave-reset policy reference-safe**. Also the current counts are padded physical rows, not counts of live routed rows after `is_padding` masking.

## 4. Independent G3, chunk and q1 incompatibility

Semantic CPU evidence `semantic_receipt.json`:

- full6144 position6128 begins experts96–103 because6128*8%352=96. Split D16 reset begins0–7. **All16 tail token rows differ**.
- Prefix chunks[2048,2048,2032,16] with reset disagree with full6144 on4096 token rows. Absolute-position IDs agree exactly for every real row, independent of chunk splits.
- q1 continuation positions6144,6145,6146,6147 begins experts224,232,240,248; per-forward reset repeats0. A real q1(position6144)+dummy0 produces experts224–231 and0–7, not Hamilton's0–15.
- Two requests at the same logical position6144 repeat the same eight experts (max count2). They preserve request/PD/reference invariance but **not per-wave balance**. Adding a deterministic per-request phase does not guarantee exact balance for arbitrary future batches and requires stable cross-role/reference identity.
- This conflict is fundamental even without padding: current ascending-Hamilton for every single-token wave requires experts0–7 for every token, while a two-token wave requires0–15; grouping invariance and that exact policy cannot both hold for arbitrary batches.

### Real/dummy/padding contract

Current default `VLLM_MOE_SKIP_PADDING=1` is unchanged (`envs.py:1655`). Ordinary biased router passes existing `ForwardContext.is_padding` to native `topk_sigmoid` (`fused_topk_bias_router.py:28–31,56–74,228–238`); grouped-router source explicitly zeros weights/uses IDs-1 on padding. Existing random simulator has no equivalent padding mask. Any approved extension must preserve the padding behavior; generating N*K ordinary routes for masked rows is a change, not neutral overhead.

Actual MRv2 `InputBatch.make_dummy` source executed on CPU confirms all-zero positions and mask-all-padding (`gpu/input_batch.py:125–215`). Real positions are prepared only for actual request rows (`input_batch.py:347–381`); unused padded positions may be stale. Stable-position routing must mask padding, never infer a real position from stale/zero rows, and must not let a dummy/DP graph bucket shift another request's routing. Hypothetical real-row IDs are invariant to adding dummy/stale-position rows, but physical vs live-routed count conservation must be defined separately. No claim is made that capture01's dummy ran expert GEMMs; only its physical participation/row count is retained.

## 5. Stable offset availability without extra reads/collectives

Frozen MRv2 carries exactly what a reference-safe policy needs, **but not at the existing router interface**:

- `gpu/states.py:59–65,113–115`: native device `num_computed_tokens`; CPU mirror is explicitly optimistic upper bound. Do not assume this CPU mirror is exact during async/speculation.
- `gpu/input_batch.py:347–381`: device positions are computed as exact native `num_computed_tokens+local_query_offset`; `gpu/model_runner.py:1487–1494,1538–1566,1986–1993` passes the existing positions buffer into the model. This is no new data producer, device readback or collective.
- `models/step5/nvidia/model.py:2013–2060`: decoder receives positions for attention but `_forward_ffn`/Step5MoE currently receives only hidden states/padding; Step passes no position/input-ID metadata to MoERunner (`1809–1823`). Existing router `input_ids` means vocabulary token IDs, not positions; do not overload it.
- `ForwardContext:168–216` carries DP metadata/is_padding but no routing positions. Attention builders sometimes retain positions, but reading a layer-dependent private DSA metadata object is not a supported all88-layer router contract.
- FULL graphs replay no Python model/router body (`gpu/model_runner.py:2024–2034`). A Python per-forward offset/counter is therefore insufficient. An approved extension must bind routing to the **already-updated stable device positions buffer at graph capture**, also on eager/PIECEWISE, and use tensor-dependent operations during replay (`gpu/cudagraph_utils.py:683–688,744–759`). Exact current positions can be consumed without new GPU readback or synchronization.

## 6. Materially different options and smallest complete scope

### A — Exact current Frontier per-wave Hamilton

Use existing selector with a newly registered native strategy; global DP prefix from existing CPU physical counts and tensor cyclic IDs. No expert backend change. Cost: small native routing/context/config/test change plus new image/package; GPU routing operator identity changes. **Cannot pass original independent full-local G3 under arbitrary split/chunks/q1**. There is also a separate padding-count decision: matching Hamilton over all physical rows would route currently masked padding, whereas preserving native skip-padding requires Frontier/live-route counts instead. Existing padded DP counts alone do not reveal each lane's live-route count. Neither routing masked padding nor adding count coordination is approved. Thus A is not an immediately admissible packet; any separate timing-only workflow and its weight/mask semantics require explicit authorization. Do not quietly weaken qualification or reference.

### B — Request-position deterministic cyclic policy (recommended only if YC approves semantics)

`ID(position,j)=(position*K+j)%E` over logical expert IDs. For each request, contiguous44-token cycles balance all352 experts exactly; any contiguous interval has max-min<=1 but remainder is rotated rather than always ascending0. Same position produces same distinct eight IDs independent of DP placement, P/D split, prefix chunking and graph padding. Arbitrary multi-request/short waves can be imbalanced. This is a **changed Frontier policy**, not the existing Hamilton label.

Minimum complete implementation proposal after approval, not performed:

1. Native `router/routing_simulator_router.py:RoutingStrategy/registry/Router` and `router_factory.py:120–126`: register the approved position policy under the **existing env selector**; preserve Step scoring_func, renormalize and scaling in the router/strategy interface. Compute selected weights from unbiased sigmoid scores, normalize over selected experts, scale3 via model settings (not literals). Correction bias no longer chooses IDs by definition; gate GEMM still runs. Preserve existing random/normal behaviors when not selecting the new mode.
2. Native `forward_context.py:ForwardContext/set_forward_context`, `gpu/model_runner.py:2042–2054`, `gpu/cudagraph_utils.py:744–753`: explicitly bind optional routing_positions to existing input_batch/input_buffers positions in eager/PIECEWISE/FULL capture. Approved mode requires correct scope and mask; no fallback/default-to0 when absent. No added readback, CPU-position approximation, counter, collective or backend change. Alternate explicit tensor argument through the model/runner/custom-op ABI is larger; use it only if required by actual graph/compile tests.
3. Native focused routing tests for both modular precisions, real/dummy masking, exact IDs/weights, chunk/PD/q1 invariance, allTP agreement and graph replay dependency. Existing CPU method checks do **not** substitute for GPU graph validation.
4. Frontier parent-owned `moe_ep_workload.py:774–855` and existing producer of layer-wave routing inputs must materialize sums over actual per-request logical token intervals plus agreed routed padding policy. A global scalar N and equal1/E alone cannot represent B. Use existing request progress rather than introduce a duplicate scheduler. Preserve ownership split and distinguish physical rows vs live routes. Parent selects its precise producer seam; this child does not touch that stream.
5. Frontier `profiling/moe/moe_impl.py:317–335`, existing MoE profile metadata/producer: new routing identity and actual kernel receipts. `standard_fused_topk` means logit-based top-k; `uniform_topk` means equal weights. Neither may be relabeled as the new cyclic+sigmoid policy. Keep unaffected expert/communication rows only where their actual shape/kernel/routing-count prerequisites still hold. Route time/op-gap must match the new operations; no CPU-overhead campaign now.

The extension remains a synthetic expert-choice policy, not original-model semantic output. Independent G3 can compare two native executions with the **same approved policy**; that is not permission to assume state equality. Cross-shape FP8/DeepGEMM/Triton/attention numerical differences remain native evidence to diagnose; no tolerance or backend substitution is approved.

### Image/source/package and primary live checks

Both options change native behavior. Parent must approve exact policy, weight/mask semantics and changed locked reference **before implementation**. Build/approve one immutable derived image/source checkout and update actual native source pins/origins (retain unaffected pins; revise affected ones and include newly imported routing files). Do not mount a hidden monkey patch over the frozen image or claim all50 pins remain unchanged after a native extension. Capture03 and originals stay immutable; create **capture04 or another unused generation** after integrated bytes are final. Recompute canonical builder's import origins and source manifest, approved routing config and actual engine/env identities. No package built here.

Before actual submission primary must verify: installed FP8/BF16 modular classes/auto oracle and E8M0 condition; effective new selector/router mode on both roles/all16 EP ranks; native request positions after P6128 transfer, D16/full6144 and q1; allTP identical IDs; padding mask/count agreement; FULL graph replay consumes updated positions; expected selected weights; independent G3 unchanged criteria; actual new routing kernels/op-gap identity; resource/creator/mount/D32000/P-auto/90min/48GPUh limits. Auxiliary diagnostic validators remain soft gates at>=32GPU, but missing routing proof means no balanced qualification. No job/attempt/submitted marker was created.

## Verification and remaining unknowns

Commands and environment are in `test_report_2026-10-22_native_routing_contract.md`. `cpu_receipt.json` and `semantic_receipt.json` are PASS for stated CPU method/math checks only. Source and retained native module evidence settle reachability; **new GPU/operator/graph/G3 agreement remains UNVERIFIED**, and user semantics/native extension approval are pending. Stop at this decision; primary continues independent SWA/identity integration. Workflow -> GPU/operator-gap alignment -> CPU overhead last.
