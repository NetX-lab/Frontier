# D29 position_cyclic native implementation design

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Scoped the approved position policy to four native files, two focused tests and explicit derived-image/package integration; design only. |

## Status, authority and counting convention

TaskID: **D29-NATIVE-BALANCED-DESIGN**. This deliverable is a concrete implementation design, not implementation. YC's position-cyclic approval was read from the primary worktree `/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/requirements.md:190–214` and `/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_d29_native_balanced/requirements.md:30–48`. The fast-forward target still contains this worktree's historical27-line D29 requirements and old proposal-only plan; the newer primary working-tree approvals and current scoped user instruction take precedence. Those files/proposal/old receipts are not edited here. Owned branch was clean and fast-forwarded from `efcd7a0` to `fa5bae2980366da1d075125d6c26248a737fc419`. Only this document is changed.

Native read-only root **N**: `/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm`. Frontier root **F**: `/data/ycfeng/Frontier/.worktrees/step5-d24-stage`. Production implementation goes into a **new derived-source candidate**, never N. All native paths below are relative to that candidate's `vllm/`; F paths retain their repository prefix.

Bounds count **added plus deleted diff lines**, including comments/imports/tests, not net additions. They are planning ceilings, not measured final diffs. Native source + tests is **6 files / 438–695 changed lines**. Derived-image/package code + tests adds **9 files / 260–420 lines**, plus **1 Dockerfile / 18–28 lines**. Total this native-facing correction, including integration, is **16 code/test/packaging files / 716–1143 changed lines**, before parent-owned Frontier/profile files. Generated native patch/snapshot, source receipts and this design are not a substitute for counting the four actual native files. This explicitly exceeds both the >5-files and >500-lines repair thresholds; parent must obtain one numerical scope approval for the combined correction before production edits. No attempt to split one correction into artificial smaller repairs is proposed.

## 1. Binding decision: use Step's uncompiled outer text-model call

**Recommended minimal correct seam:** add a typed optional `ForwardContext.routing_positions` and a small scoped setter; enter it in `Step5ForCausalLM.forward` around the existing `self.model(...)` call **only when the existing selector chooses `position_cyclic`**. Keep the existing ForwardContext object, DPMetadata, attention metadata, padding tensor, MoE layer counter and additional kwargs unchanged. Save/restore only the prior positions reference in `try/finally`; do not replace the whole dataclass or recreate `set_forward_context`, which can coordinate DP and reset state.

Illustrative interface, not a patch:

```python
@contextmanager
def set_routing_positions(positions: torch.Tensor):
    context = get_forward_context()
    previous = context.routing_positions
    context.routing_positions = positions
    try:
        yield
    finally:
        context.routing_positions = previous
```

At `Step5ForCausalLM.forward`, preserve the old call exactly for all other modes; the selected path uses this setter with the **actual `positions` argument**, not vocabulary IDs, request IDs, a CPU computed-token mirror, a global counter or per-request phase. The setter is not a tensor producer: no clone/copy/readback/all_reduce. The native router reads that tensor plus existing `context.is_padding` inside its ordinary modular `select_experts` call.

### Why this is correct for the frozen runtime

| Runtime | Source evidence | Binding behavior |
| --- | --- | --- |
| Eager/NONE | `gpu/model_runner.py:1986–1993,2042–2071` passes `input_batch.positions` under the existing context. | Outer text-model Python executes each call; scoped tensor reference covers all routed layers and is restored on normal/exception exit. |
| PIECEWISE | `config/vllm.py:100–123,770–776,814–842` includes both Step architectures in default **breakable** CUDA graphs on CUDA and sets compilation mode NONE. `gpu/cudagraph_utils.py:189–193,565–578,625–639` chooses this existing path. | This is **not a compiled FX Step model**. Breakable capture executes the outer text model once and captures MoE routing tensor ops in graph segments. `breakable_cudagraph.py:369–405,417–435` replays segments, not model Python; routing kernels keep input-buffer addresses from capture. No replay-time setter is required. |
| FULL | `gpu/cudagraph_utils.py:683–688,744–759` passes stable `input_buffers.positions[:num_tokens]` under context during warmup/capture; `:491–497` invokes the forward with runtime mode NONE inside actual FULL stream capture. | Outer setter binds the stable device slice **during capture**. Do not branch on context mode FULL: the capture forward's context mode is NONE. Routing tensor operations capture pointer dependencies. `gpu/model_runner.py:2024–2033` replays after native input staging, with no Python routing body. |

Exact positions already come from `_prepare_pos_seq_lens_kernel` (`gpu/input_batch.py:347–381`), using native device computed-token counts. FULL/PIECEWISE input buffers are updated by existing input preparation. Padding positions can be stale; the existing mask must invalidate those rows regardless of their positions.

`Step5ForCausalLM.forward` (`models/step5/nvidia/model.py:2403–2410`) and `Step5Model` are not `support_torch_compile` decorated. The VLM reaches that text-model call via `Step3p7...:70–76` and `Step3VL...forward`; no change to VL/frontend/decoder FFN argument lists is needed. Both MoE methods cross the existing opaque MoE op (`runner/moe_runner.py:198–214,309–327,737–746`); the router executes within it before DP gather (`:627–642`, `prepare_finalize/naive_dp_ep.py:136–208`). Routing is not an attention eager-break callback. Thus the binding is scoped for the whole capture traversal, while replay uses the recorded device operations.

**Supported scope:** frozen CUDA/H800, TP8/DP2/EP16/PP1, no SP/mega/DBO/MTP, existing breakable PIECEWISE + FULL. Do not enable breakable, flip backend or add a compile decorator to make this design work: it already is the native configuration. An explicit opt-out/other compiled Step runtime is not an alternative fallback; missing supported configuration is a clear preflight failure. If real graph tests reveal the scoped setter is insufficient, stop and update the file/line authorization—do not silently add runner/cudagraph edits.

### Rejected alternatives

- Add positions to `gpu/model_runner.py` and `gpu/cudagraph_utils.py`: viable but redundant for this actual uncompiled Step path; would add two core files/call sites and more propagation tests without gain here.
- Mutate `ForwardContext.additional_kwargs['routing_positions']`: fewer file edits but an untyped new dictionary convention shared between distant callers; not a better interface, and not a reason to evade the threshold.
- Bind inside `Step5DecoderLayer._forward_ffn` or thread positions through every MoE/custom-op ABI: unnecessary duplicate per-layer scope or larger interface.
- Python offset/counter at replay: invalid; graph replay bypasses the Python body. CPU optimistic counts are not exact device logical positions.

## 2. Native routing implementation, weights and mask

Use **one existing configuration mechanism**, `VLLM_MOE_ROUTING_SIMULATION_STRATEGY=position_cyclic`. Add one entry in the existing `RoutingSimulator` strategy registry, not a new CLI flag, sitecustomize registration or router override. Keep `uniform_random`, `normal_routing`, ordinary selector-empty routing and their current behavior intact. Keep `RoutingMethodType.Simulated` truthful for the new synthetic choice; do not pretend it is original biased top-k. The H800 modular oracle remains untouched; static/live tests must verify this routing classification still selects the same FP8/BF16 classes.

Minimal extension to the existing strategy interface: keyword-only `positions`, `renormalize`, `routed_scaling_factor` forwarded by `simulate_routing`; old distribution strategies accept these but do not use/change them. The new position strategy consumes them. `RoutingSimulatorRouter` receives configured `scoring_func`, `renormalize` and scaling from its existing factory call; forwards positions only for the selected position mode. Reject unsupported scoring when selecting the new mode rather than implement unrelated softmax/hash/grouped routing. Step's supported scoring is sigmoid, top_k8/E352 from model config, not literals in the new code.

New strategy algorithm (all operations device-side, with CPU tensors supported in tests):

1. Require an existing scoped tensor of shape[M], signed integer dtype and same device as logits, with logits[M,E] and `1<=K<=E`. These are static shape/device checks; do not `.item()`, `.any()` or copy tensor contents to host. Do not substitute zero positions when absent. Existing callers determine whether mask=None means all-real; MRv2 serving provides its mask.
2. Compute gather indices in int64: `ids = (positions[:, None] * K + arange(K)) % E`. For native serving use actual configured E/K. No per-request phase, wave offset, layer rotation or expert-ownership change.
3. Gather **unbiased** selected FP32 logits, sigmoid, normalize by selected score sum only if configured, then apply configured scale. Return FP32 weights and IDs in the requested native index dtype (native ordinary default is int32). Selection bias remains an upstream model parameter but does not change deterministic IDs or selected weights. Do not add epsilon/clamp, arbitrary tolerance, weight=1 shortcut or fallback. Use the same selected-score/normalize/scale contract shown by locked `fused_topk_bias_router.py:311–317,362–370`; CUDA arithmetic agreement must be measured, not inferred from a CPU formula.
4. Preserve `VLLM_MOE_SKIP_PADDING` and existing context-mask semantics (`fused_topk_bias_router.py:28–31,56–74`). After computing the valid gather indices/weights, invalidate masked rows with IDs=-1 and weights=0. Do not gather -1 before masking, do not route dummy rows, and do not use Python boolean checks on device masks. If skip-padding is explicitly off in a supported standalone test, maintain native behavior and label that case; frozen serving keeps it on.
5. Existing BaseRouter maps/records logical IDs and converts dtype (`base_router.py:260–305`); EP ownership and dispatch/combine remain unchanged. Frozen EPLB is off. No new distributed operation or per-forward metadata readback is introduced.

Preserve auto oracle and the actual FP8 **E8M0 OR valid-shape** subkernel condition; small M does not imply Triton. Keep GateLinear, shared experts, activation/swiglustep, native IO/training failures and all G3/transfer criteria untouched. `position_cyclic` is a synthetic policy with consistent native reference semantics, not original-model semantics and not a numerical-state waiver.

## 3. Exact file/line scope for numerical authorization

### A. Native behavior and focused tests: 6 files, 438–695 changed lines

| # | Exact file | Current edit seam | Bound | Gain / acceptance |
| --- | --- | --- | --- | --- |
| N1 | `vllm/forward_context.py` | field after is_padding:181–184; setter after get_forward_context:227–233 | 25–35 | Typed scoped routing tensor; preserves existing context/counter and nested/exception cleanup. Do not edit create/set_forward_context or DP coordination. |
| N2 | `vllm/models/step5/nvidia/model.py` | import:35; constructor of Step5ForCausalLM:2347; outer forward:2403–2410 | 20–30 | Selective scope around the already-passed positions tensor; no decoder, runner, frontend or graph-mode change. |
| N3 | `vllm/model_executor/layers/fused_moe/router/router_factory.py` | simulation constructor:120–126 | 8–15 | Forward configured scoring/normalize/scale; selector/priority/old router branches unchanged. |
| N4 | `vllm/model_executor/layers/fused_moe/router/routing_simulator_router.py` | strategy interface:17–40; distribution interface:85–91; registry:226–235; simulate:258–298; router:304–337 | 125–185 | One stateless registered position strategy and explicit tensor/weight/mask forwarding; no compiled extension/new kernel implementation. |
| T1 | F `tests/unit/test_step5_native_position_cyclic.py` **new** | Reuse source-method/seam fixtures from existing native-source tests; bounded file | 140–210 | CPU actual derived-source method tests, ID/weight/mask contracts, full/split/chunk/q1 invariance, old-mode/factory controls and source route assertions. |
| T2 | F `tests/integration/step5_environment/test_position_cyclic_graph.py` **new** | Actual imported derived-native router/context + thin Step outer-forward fixture; pytest GPU markers | 120–220 | Actual CUDA NONE/breakable PW/FULL replay under changing positions/mask; minimal native method/oracle census on both precisions. No engine submission code or host-overhead suite. |

Native production subtotal N1–N4: **4 files /178–265**. Tests: **2 files /260–430**. No csrc, CUDA C++, Triton expert/oracle, FusedMoE layer/custom-op ABI, `gpu/model_runner.py`, `gpu/cudagraph_utils.py`, `model_state.py`, `vl.py` or envs.py edits planned. Tests live in F because the exported image source has no native tests checkout; this does not conceal native files in the test file count.

### B. Derived-image/package integration: 9 code/test files, 260–420 changed lines; 1 Dockerfile, 18–28

These are required by inspected hard locks, not optional tooling polish. Generated artifacts are additional deliverables, not counted as code edits. Parent owns their implementation/publication, but must include this scope in its consolidated numerical approval.

| # | Exact F file | Edit seam / purpose | Bound |
| --- | --- | --- | --- |
| P1 | `tests/integration/step5_native_image/build_image.py` **new** | Exclusive build-context preparation only: verify unchanged base hashes, validate four-file allowlist/diff, copy approved derived bytes, generate image context/pins/manifest; no API/auth/submit/push code. Avoid existing broad overlay's destructive-copy behavior. | 80–120 |
| P2 | `tests/integration/step5_native_image/Dockerfile.position_cyclic` **new** | FROM exact base digest; COPY only the four authorized native files into installed package; verify installed hashes/compile without GPU; no pip install/compiler/backend rebuild/forced routing ENV baked into default image. | 18–28 |
| P3 | `tests/comparison/dp_placement_pp/vllm_replay.py` | Existing RUNTIME_ADAPTERS:178–190 and server_env:312–335: add one derived Step adapter using existing Step metadata/connector/version and exact derived-image/base/policy identity; set existing simulation env from that adapter, include it in returned effective env, remove inherited contamination. Do not change old adapter/image or add a second routing config mechanism. | 35–55 |
| P4 | `tests/integration/step5_pd/r2_canonical.py` | definition/CLI:16–20,133–149: accept explicit registered runtime label for new generation while keeping historical default and frozen engine/resources/waves/criteria; no synthetic workload replacement. | 10–20 |
| P5 | `tests/integration/step5_pd/r2_stage.py` | guard:37–38 and schema pin formation:45–50: use existing adapter's Step architecture classification rather than exact historical label; include three added native route modules with model module already bound. | 8–15 |
| P6 | `tests/integration/step5_pd/r2_runtime.py` | SOURCE_MODULES:8–17; bind_model_state:84–99: pin actual imported forward_context/router_factory/simulator modules and emit one deduplicated model/router configuration/backend census per role/rank via existing observer; all88 layers accounted for, no tensor contents/readback in serving. | 35–55 |
| P7 | `tests/integration/step5_pd/verify_r2_stage.py` | image hard lock:36; env checks:115–117: require exact declared adapter/digest/base/policy, same both-role selector, correct new pin set/source origins and unchanged D-only32000/P-auto/budget. Keep old packet controls exact. | 20–35 |
| P8 | `tests/unit/test_step5_canonical_g3.py` | existing stage_capture/golden controls:15–99: new derived adapter/policy and pin/env cases, old no-override baseline and D-only control; rejects wrong image/policy/role/origin. | 45–70 |
| P9 | `tests/unit/test_step5_native_pd_r2_observation.py` | actual runtime/native-source census fixtures: add configured mode/origin/both-backend and observer-failure soft-gate checks using existing fixtures | 15–30 |
| P10 | `tests/unit/test_step5_native_image.py` **new** | Context builder accepts only four source files, byte-validates base and new source hashes, refuses existing context/foreign changes; container/image operations mocked/not run locally | 12–20 |

P1/P3–P10 total **9 code/test files /260–420 lines**. Numeric ranges are checked sums; P2 is explicitly another file, not hidden in the script. Proposed derived adapter label `step5_position_cyclic` is a runtime identity entry, not a second routing strategy name; canonical policy remains `position_cyclic`. Adapter `native_reference` owns the immutable derived image, `base_image` and `routing_strategy`, and existing exact `check_native_reference` checks them. Reuse Step base metadata/connector classification; the next derived generation needs one identity entry, not a new family condition spread through callers.

**Not owned/countable here:** parent Frontier logical-position workload/count/profiler semantic and profile-input schema/gate changes. Parent adds its independently designed file/line totals once. Existing clean-canonical or profiling launchers have their own old-image locks; they cannot be declared ready by this D24 package scope and belong in parent's profile/acceptance design. Avoid rewriting all historical JSON profiles or frozen sources. Derived adapter uses same installed vLLM version but **different digest and source bytes**; version equality is not source identity.

## 4. Verification strategy: CPU versus real GPU

### CPU before image build

- Compile new native files and source-method tests with Python3.12; exact E352/K8 positions0,43,44,6128,6144 and supplied shape/dtype controls. IDs distinct within each unmasked row; conserved8 routes per valid row, zero per masked row. Contiguous44 tokens cover each expert once; repeated-position requests intentionally repeat experts.
- Compare split[0,6128)+[6128,6144) and chunks[2048,2048,2032,16] with independently evaluated full[0,6144) IDs; q1 positions6144..6147 begin224/232/240/248. Padding with stale positions does not change valid IDs; masked rows=-1/0. Do not assert arbitrary multi-request waves max-min<=1.
- Weight controls use **same supplied logits** and selected unbiased sigmoid values, normalize on/off and non-default supplied scale, not weight=1. Preserve old random/normal outputs given their existing seeded controls and ordinary selector-empty factory. Missing positions/device/shape fail with explicit errors; no fallback/readback.
- Actual Step outer-forward CPU seam checks tensor **identity**, scoped cleanup/nested invocation and preservation of the same ForwardContext/MoE counter. Source/actual config controls establish default breakable mode and no compile decorator. CPU contract probes alone do not prove CUDA graphs or numerical G3.
- Reuse existing staged collector tests (D24 raw/mean, pressure actual pre-grant demand, HTTP, startup), source/hash/origin verifier, bash/compile/node preflight and offline ResourceConfig/dry-run. Never instantiate RJobBackend as part of offline validation. No CPU-overhead model/sampling suite.

### Parent-owned bounded GPU checks in derived image

T2 has no submitter/backend/auth code. Run it as a bounded test in parent's approved worker, not local GPU. Import actual installed derived native modules from neutral cwd, not source shadowing. Capture **all-padding dummy** first, then update the same device positions/mask buffer and replay with real q1/tail rows: this catches the particularly dangerous false success where capture-specialized all-padding routing never becomes live.

- CUDA eager; **actual BreakableCUDAGraphWrapper** PIECEWISE with current descriptor; FULL `torch.cuda.CUDAGraph` capture under the same context mode NONE used by native FULL capture. Two replays with positions/mask changes, no Python setter/model/router call on replay. Match output IDs/weights to independent expectation; unchanged addresses. Nested context/failure restoration check runs outside capture. Test-only bounded output checks may synchronize **after execution**; no readback/collective is added to production routing or timed serving.
- CPU platform direct-MoE seam is insufficient: exercise the existing opaque `torch.ops.vllm.moe_forward` route through a thin real Step forward fixture for both selected modular methods, with actual router and deterministic output sink. A tiny fixture proves binding/capture semantics, not full expert math.
- Reuse retained native dispatch-module test mechanism for real FP8/BF16 expert modules with `moe_backend=auto` and frozen TP8/DP2/EP16. Record `RoutingMethodType.Simulated`, quant method, prepared expert class, E8M0 decision, actual executed expert class and route kernel names. Do not monkeypatch oracle to Triton and call it native proof. CPU class/source checks + fixture may be insufficient for all16-rank actual dispatch; parent must retain full module/serving evidence.
- Full native package: P6128+D16 versus independent Dfull6144 reference, chunk/boundary/q1, real+dummy/padded lanes, allTP agreement, FULL runtime use, bit-exact transferred prefix and current independent-state criterion, no copied actual top-k. The existing D24 diagnostic adds state values and D32000 pressure evidence; missing numerical state agreement stays UNVERIFIED. All>=32GPU auxiliary checks are soft gates; native IO/training failure handling unchanged.

**Scope stop rule:** if these tests require a new routing kernel, expanded custom-op interface, runner binding, extra readback/collective or additional code files/lines, report a new design/limit before editing. Do not choose a backend/configuration that merely avoids the failing actual path.

## 5. Interface to parent profiler and evidence identity

Profiler uses the **same native factory/router**, not a parallel implementation of IDs or weights. Set `VLLM_MOE_ROUTING_SIMULATION_STRATEGY=position_cyclic`, create router with the layer group's configured sigmoid/normalize/scale, bind the supplied device logical positions via `set_routing_positions` inside its existing native ForwardContext and mask. Call `select_experts` with requested native dtype and use its outputs as expert inputs. Gate GEMM stays measured separately as currently; the route measurement must actually include the new tensor operations.

Parent must add profile metadata/selection identity for `position_cyclic`, with logical position intervals or exact test positions, **physical rows, valid routed rows and expert-token counts separately**, configured scoring/normalize/scale/skip-padding, FP8/BF16 layer group, expert backend/actual kernel/E8M0 and source/image identity. Preserve old `standard_fused_topk` (logit-selected) and `uniform_topk` (equal-weight) rows; neither identifies this path. The new implementation uses ordinary Torch tensor kernels inside the native opaque MoE op, not the old fused native `topk_sigmoid`; actual device names/launch counts may differ and require new route/op-gap timing. No fabricated kernel count/cost, relabelled historical profiles or CPU-overhead measurement now.

`BaseRouter.set_capture_fn` already permits logical-ID capture for **test-only** checks; do not turn it into a production host callback with `.cpu()` during routing/capture. The static r2 runtime census proves configured mode/source/backends only, not replayed ID values. Real graph contract tests/full serving evidence close that gap; label the evidence tiers honestly.

## 6. Immutable source/image/package recipe (parent executes later)

1. Create a new candidate directory under `/data/ycfeng/tmp`, exclusively new; bind base exported bytes to immutable base image `a.z01.normandyfunc.com/docker-public/vllm/vllm-openai@sha256:69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20`. Keep N/captures01/02/03 unchanged. Canonical candidate contains exactly N1–N4 edits; all unrelated native bytes unchanged. Store reviewed unified diff + four actual changed paths and before/after SHA256, not one opaque patch counted as one code file.
2. P1 creates a fresh build context containing P2 and an allowlisted `vllm/` tree of those **four** edited source files. Include generated `base_source_sha256.json`, `derived_source_sha256.json` and `derivation_manifest.json` (base digest, patch hash, path list, source IDs, no invented derived digest). No symlink traversal, no original-folder overwrite, no broad overlay.
3. P2 FROM exact locked base; COPY authorized files to `/usr/local/lib/python3.12/dist-packages/vllm/...`; retain native `.so`, Torch/CUDA/DeepGEMM wheels, envs and entrypoint. Compile/hash those files with image Python without loading GPU. Do not COPY the whole image_src tree or use pip install/recompile to introduce unrelated native changes. Python bytecode is verified/invalidated normally during image construction; do not leave stale sourceless bytecode masquerading as the new implementation. Do **not** bake `position_cyclic` as a global image ENV that silently changes other router modes: the selected adapter sets effective serving env explicitly.
4. Parent builds/publishes internal image only after consolidated scope approval and code/test review, via company registry handbook. Example target namespace: `hub.stepfun-inc.com/vllm-openai/step5-position-cyclic:<new-unique-tag>`; never Docker Hub. Use the authorized target's actual repository permissions; no registry auth/inspection/build/push attempted here. Verify remote derived linux/amd64 digest and exact source/compiled-library/dependency identities; unlike an unchanged retag, **derived digest must differ** from base. Do not overwrite an existing different tag.
5. Before GPU submission, parent exports actual installed **derived** native sources and a fresh `installed_source_pins.txt` from that image. Of the four edited paths, only `models/step5/nvidia/model.py` occurs in the original50 listing; the three new imported route/context paths are appended. At minimum expected native set is original50 with the changed model pin updated plus `forward_context.py`, `router/router_factory.py`, `router/routing_simulator_router.py` = **53 installed pins**. Verify count from actual allowlist, not blind50. Bind new module origins to installed root and hashes and checkout bytes; a nested manifest hash or unchanged vLLM version alone is insufficient.
6. After code integration and image digest are final, use the **existing canonical G3 builder** with new adapter/pins and new unused generation (capture04 if unused; otherwise next unused), same P/D workload/topology/ports/API/counts/context/128seq/runtime/criteria, P automatic pool, D32000 including independent reference, H80032/codesign nonpreemptible,90min/48GPUh. D24 collector raw/mean and original pressure/ownership/READ checks remain. Cloud path is an approved personal `/mnt/codesign-exp/ycfeng/...` string until parent operates it. Capture03 stays sealed/unsubmitted/not balanced.
7. Exact effective env for both roles must include selector `position_cyclic`; no extra request phase/layer rotation, no old random selector inherited. Validate image identity on node preflight, actual imported origins on all ranks, source/config/tool hashes after integration, ResourceConfig/dry-run offline and fresh creator/mount/GPU/live evidence by parent. Keep launcher alive during any later platform job. No submitted/job marker is created by this design.

Thus 53 total is the intended minimum, not 50 unchanged or 54. Other inspected read-only route/oracle/runtime sources remain provenance witnesses and can be retained in derived-source receipts without pretending they were code changes.

## Design validation / unresolved boundaries

Observed locally: clean fast-forward to fa5bae2; actual locked configuration enables breakable Step PW and disables torch.compile; FULL captures with context mode NONE; both modular routes already accept precomputed top-k; existing stage/image locks require the explicit integration files above. A bounded read-only hash/listing check confirmed capture03 manifest `cdaad24ecded6da6f2eea90ed4ef4a582432db2fd75c56c29aafb22f2f56a57d` unchanged, original50 pin paths and the53-pin minimum. No new routing tests, GPU/engine/image commands run in this design phase; prior CPU/GPU receipts are historical and not claimed for the new design. No native/prod/proposal/approval/frozen packet bytes edited.

Native design is ready for the parent's **combined numerical scope approval**. Remaining live questions (graph replay pointer dependence, actual auto oracle/E8M0 subkernel, exact selected-score CUDA behavior, full G3 state agreement and actual new profile/op-gap timing) are verification items, not reasons for speculative additional code or immediate wider research. Proceed workflow -> GPU/op-gap -> CPU overhead last.

## 1. 规划完成情况

**汇总：** PASS 4｜BLOCKED/FAILED/SKIPPED 0｜TODO 0（仅本次 design scope）

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 实际 Step binding／graph 路径 | PASS | 本文§1，已有 positions 与 breakable/FULL capture 源码。 |
| 最小文件与数值范围 | PASS | 本文§3，16 files／716–1143 changed lines，未拆分规避批准。 |
| CPU／GPU 验证与 profiler 接口 | PASS | 本文§4–5，GPU 结果仍待实际实现验证。 |
| 派生 image／新 package 设计 | PASS | 本文§6，原包和源只读，未构建或发布。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 本次 bounded implementation-design 请求至今｜**当前前沿：** 提交本文并交回主会话合并范围。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 只设计下一步所需 native/image 接口，未扩展 harness。 |
| D2 细节反复重试 | CLEAR | 无实施修复或重试；graph 源码调查给出明确最小 seam。 |
| D3 支线问题修补 | CLEAR | 全部检查服务 binding／范围／封存要求。 |
| D4 可并行任务停滞 | CLEAR | 用户明确本子任务不再 delegation；父端独立设计未被阻塞。 |
| D5 超范围镀金 | CLEAR | 仅此 design 文件，无 production/native 改动。 |
| D6 分析空转 | CLEAR | design 即本次交付，调查形成文件表与决定。 |
| D7 路线违背 | CLEAR | 未执行 image/GPU/platform 或 CPU-overhead 工作。 |

**文档检查：** S5：本工作树 D29 requirements／plan 仍为历史 proposal-only；主工作树 D29 `progress.md:18` 仍写 decision pending，且 `plan.md:20–24` 未标为历史。较新的 approvals 与 `plan.md:12–18` 已明确当前路线。S1–S4／S6 未发现明显缺陷。
**归因：** 无执行偏离。

## 3. 纠正建议

**分支：** B：旧记录存在缺陷，主会话更新历史状态后继续。

**文档更新（先执行）：**
1. 主工作树 D29 `progress.md:18`：记录已批准 policy 与 design 完成，待 combined numerical scope／实现／live evidence；`plan.md:20–24` 标为历史阶段。（可直接执行，父端所有权；本子任务不能编辑。）

**需停止的偏离行为：** 无。

**接下去的路线：**
1. 本子任务提交本文 → 完成标志：仅本文 commit，branch clean。
2. 主会话并行汇总 Frontier／profile 设计 → 完成标志：同一 semantic correction 的完整文件／行数范围。
3. 一次范围批准后再实施 → 完成标志：reviewed source 与 CPU／GPU／image／new-generation 证据；不得提前宣称 G3 成功。

## 4. 最值得关注的 3 点

1. 最小 native source＋tests 已是6 files，需整体数值范围批准。
2. 既有 device positions 可用，但真实 graph replay／weights／auto-backend／G3 尚未验证。
3. 主会话保持 workflow → GPU/op-gap → CPU overhead，原 capture03 不得重标 balanced。
