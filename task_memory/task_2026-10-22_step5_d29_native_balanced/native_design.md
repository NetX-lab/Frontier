# D29 position_cyclic native implementation design

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Link the explicit case-local post-build full-source receipt command required by profiling; no native/builder code allowance change. |
| 2026-10-22 | Scoped the approved position policy to four native files, two focused tests and explicit derived-image/package integration; design only. |
| 2026-10-22 | Addressed independent review's four P2 gaps: actual client adapter/HTTP capability, imported model provenance, genuine eager-break replay and correctly bound native-module evidence; revised unique-file/line budget. |

## Status, authority and counting convention

TaskID: **D29-NATIVE-BALANCED-DESIGN**. This deliverable is a concrete implementation design, not implementation. YC's position-cyclic approval was reread from primary `/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/requirements.md:191–215` and `/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_d29_native_balanced/requirements.md:30–48`. Primary integrated the first design commit `651415a` and updated current authority/spec/progress at `9ed57630f2f564498f9706c7a5af0243928d1370`; its current approvals and this review-correction instruction take precedence over this worktree's historical proposal-only records. Those parent files, proposal and old receipts are not edited here. This branch was previously fast-forwarded to `fa5bae2980366da1d075125d6c26248a737fc419`; this correction starts from its clean design commit `651415a372a25150344ead8d1dc5fb9510e53acf`. Only this document is changed; no further merge is needed for this isolated correction.

Native read-only root **N**: `/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm`. Frontier root **F**: `/data/ycfeng/Frontier/.worktrees/step5-d24-stage`. Production implementation goes into a **new derived-source candidate**, never N. All native paths below are relative to that candidate's `vllm/`; F paths retain their repository prefix.

Bounds count **added plus deleted diff lines**, including comments/imports/tests, not net additions. They are estimates, not measured implementation diffs or permission to exceed the final approved ceiling. Native source + tests is **6 unique files / 558–865 changed lines**. Derived-image/package/client code + tests adds **12 unique files / 427–689 lines**, plus **1 Dockerfile / 18–28 lines**. Total this native-facing correction is **19 unique code/test/packaging files / 1003–1582 changed lines**, before parent-owned Frontier/profile files. Compared with the first design's16 files/716–1143 lines, this adds the two missing clients and one unavoidable existing caller-test fixture file, and expands already-counted provenance/graph/regression tests. Generated native patch/snapshot, source receipts and this design are not substitutes for counting the four actual native files. This exceeds both repair thresholds; parent must obtain one numerical scope approval for the combined correction before production edits. No artificial splitting or hiding native files inside a patch is proposed.

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
| PIECEWISE | `config/vllm.py:100–123,770–776,814–842` includes both Step architectures in default **breakable** CUDA graphs on CUDA and sets compilation mode NONE. `gpu/cudagraph_utils.py:189–193,565–578,625–639` chooses this existing path. | This is **not a compiled FX Step model**. Breakable capture executes the outer text model once; replay runs graph segments and recorded eager callbacks, not outer model Python. The supported routing path stays in a graph segment even **after a genuine eager break**, reading capture-bound positions/mask buffer addresses. This needs T2's multi-segment test below, not merely a single-segment wrapper. No replay-time setter is required for captured routing. |
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

### A. Native behavior and focused tests: 6 unique files, 558–865 changed lines

| # | Exact file | Current edit seam | Bound | Gain / acceptance |
| --- | --- | --- | --- | --- |
| N1 | `vllm/forward_context.py` | field after is_padding:181–184; setter after get_forward_context:227–233 | 25–35 | Typed scoped routing tensor; preserves existing context/counter and nested/exception cleanup. Do not edit create/set_forward_context or DP coordination. |
| N2 | `vllm/models/step5/nvidia/model.py` | import:35; constructor of Step5ForCausalLM:2347; outer forward:2403–2410 | 20–30 | Selective scope around the already-passed positions tensor; no decoder, runner, frontend or graph-mode change. |
| N3 | `vllm/model_executor/layers/fused_moe/router/router_factory.py` | simulation constructor:120–126 | 8–15 | Forward configured scoring/normalize/scale; selector/priority/old router branches unchanged. |
| N4 | `vllm/model_executor/layers/fused_moe/router/routing_simulator_router.py` | strategy interface:17–40; distribution interface:85–91; registry:226–235; simulate:258–298; router:304–337 | 125–185 | One stateless registered position strategy and explicit tensor/weight/mask forwarding; no compiled extension/new kernel implementation. |
| T1 | F `tests/unit/test_step5_native_position_cyclic.py` **new** | Reuse source-method/seam fixtures from existing native-source tests; bounded file | 140–210 | CPU actual derived-source method tests, ID/weight/mask contracts, full/split/chunk/q1 invariance, old-mode/factory controls and source route assertions. |
| T2 | F `tests/integration/step5_environment/test_position_cyclic_graph.py` **new** | Actual derived-native router/context + thin Step outer-forward/opaque-MoE fixture; native eager-break decorator; reuse dispatch probe config/layer/observer/backend functions | 240–390 | CUDA NONE, genuine multi-segment PW and FULL replay with updated positions/mask, stable break output and router output; explicit context/mask-bound FP8/BF16 native module evidence. No old probe edits, submitter or host-overhead suite. |

Native production subtotal N1–N4: **4 files /178–265**. Tests: **2 files /380–600**. T2's larger bound includes device/config/context setup, NONE/PW/FULL controls, a real eager-break fixture/assertions and separate native FP8/BF16 module initialization/evidence; no promise to fit that work in the previous120–220 lines. No csrc, CUDA C++, Triton expert/oracle, FusedMoE layer/custom-op ABI, `gpu/model_runner.py`, `gpu/cudagraph_utils.py`, `model_state.py`, `vl.py` or envs.py edits planned. Tests live in F because the exported image source has no native tests checkout; this does not conceal native files in the test file count.

### B. Derived-image/package/client integration: 12 unique code/test files, 427–689 changed lines; 1 Dockerfile, 18–28

These are required by inspected hard locks, not optional tooling polish. Generated artifacts are additional deliverables, not counted as code edits. Parent owns their implementation/publication, but must include this scope in its consolidated numerical approval.

Profiling additionally consumes `native_installed_sources.json`. Its precise
post-build emitter is the case-local command under "Generated native-source
receipt and emitter" in
`../task_2026-10-22_step5_position_routing/profile_source_design.md`: externally
verified final/base digest, inspected installed root and sealed derivation
manifest are inputs; one new local JSON receipt is the output. The retained
command contains51 Python lines including the container payload and runs after
publication; it is not baked into the image. It adds no maintained exporter or
P1 behavior. This command is proposed/documented, syntax-checked only; actual
image execution and source/serving-subset verification remain prerequisites.

| # | Exact F file | Edit seam / purpose | Bound |
| --- | --- | --- | --- |
| P1 | `tests/integration/step5_native_image/build_image.py` **new** | Exclusive sparse build-context preparation: reuse `build_stage.sha256`/`installed_source_hashes`, byte-check base, validate four-file allowlist/diff, copy only approved derived bytes, generate context/pins/manifest. No API/auth/submit/push code; never call the existing burst overlay's destructive whole-tree copier. | 80–120 |
| P2 | `tests/integration/step5_native_image/Dockerfile.position_cyclic` **new** | FROM exact base digest; COPY only the four authorized native files into installed package; verify installed hashes/compile without GPU; no pip install/compiler/backend rebuild/forced routing ENV baked into default image. | 18–28 |
| P3 | `tests/comparison/dp_placement_pp/vllm_replay.py` | RUNTIME_ADAPTERS:148–190/server_env:312–335: one derived Step adapter, exact image/base/policy; existing simulation env explicitly selected/returned, inherited selector removed. Add **one registry capability** `http_capture` (true on old/derived Step, false on0.10.2/0.28.0). Reuse existing Step `model_metadata`/connector/version; no repeated family test or second routing configuration. | 45–70 |
| P4 | `tests/integration/step5_pd/r2_canonical.py` | definition/CLI:16–20,133–149: accept explicit registered runtime label for new generation while keeping historical default and frozen engine/resources/waves/criteria; no synthetic workload replacement. | 10–20 |
| P5 | `tests/integration/step5_pd/r2_stage.py` | guard:37–38/schema pins:45–50: reuse adapter's existing Step architecture; freeze **four added imported modules**, including `vllm.models.step5.nvidia.model`, in schema hashes. For selected derived policy, bind declared installed native root from `config['installed_package']` in observation config; local `source_root` is only a byte-export input, not serving import root. | 18–30 |
| P6 | `tests/integration/step5_pd/r2_runtime.py` | SOURCE_MODULES:8–17/bind_model_state:84–99: add model plus context/factory/simulator; verify real imported module origin and hash under the declared installed root, including actual Step forward class/method source. Reuse observer.verify_module/provenance and soft observer error boundary. Deduplicated per-role/rank88-layer router/config/backend census; no tensor reads. | 75–115 |
| P7 | `tests/integration/step5_pd/verify_r2_stage.py` | image lock:36/env checks:115–117: exact adapter/digest/base/policy, both-role selector, explicit model+route schema hashes matching corresponding installed pins and declared root. Compare staged engine.json/config engine/definition adapter; preserve D32000/P-auto/budget and old controls. Static pin agreement is **not imported-model evidence**. | 30–50 |
| P8 | `tests/unit/test_step5_canonical_g3.py` | existing stage_capture/golden/actual HTTP fixtures: derived adapter/policy/model pins/root, old baseline/D-only controls, actual R2 caller adapter propagation for PD/direct-D and mismatch rejection. Reuse `test_step5_http_capture.exchange` for both Step adapters' fresh/reused/EOF receipts and registry-negative controls; no new HTTP fixture file. | 95–155 |
| P9 | `tests/unit/test_step5_native_pd_r2_observation.py` | Existing ModuleType/source-hash/observer fixtures: exact installed model origin succeeds; byte-identical foreign-root model or wrong spec origin fails; wrong hash/missing module fails; context/factory/simulator origins, configured mode/both backends and observer error stay soft for native calls. | 35–60 |
| P10 | `tests/unit/test_step5_native_image.py` **new** | Context builder accepts only four source files, byte-validates base and new source hashes, refuses existing context/foreign changes; container/image operations mocked/not run locally | 12–20 |
| P11 | `tests/integration/step5_pd/r2_replay.py` | run:21–25,41–42,58–61: read the **required actual args.engine_config**, compare with staged config engine/definition adapter, select once via existing `canonical_replay.runtime_version`; pass that version to local_model_config, build_app and every replay (including independent direct-D). Remove all three hardcoded step5 labels, record selected client version. | 20–35 |
| P12 | `tests/comparison/calibration/pd_proxy.py` | import:46/build_app:200–203: use existing `adapter_record(vllm_version)['http_capture']`; replace object-identity check. Actual file is calibration/pd_proxy.py, not the review's dp_placement_pp path. No transport/retry/connection changes. | 4–8 |
| P13 | `tests/unit/test_step5_native_reference.py` | existing actual reference-driver fixture:402–455, args at443–445: supply `engine_config=stage/'engine.json'` required by the real CLI. Reference success/failure/continue tests otherwise unchanged. This extra existing test file is unavoidable without introducing a speculative fallback for the incomplete fixture. | 3–6 |

P1/P3–P13 total **12 code/test files /427–689 lines**; P2 is explicitly another file. All19 paths across N/T/P tables are unique. Proposed derived adapter label `step5_position_cyclic` is a runtime identity entry, not another strategy name; canonical policy remains `position_cyclic`. Adapter `native_reference` owns the immutable derived image, `base_image` and `routing_strategy`, checked by existing exact `check_native_reference`. Reuse Step base metadata/connector classification and declare `http_capture` once per registry member. **Inspection found no existing HTTP capability field**: connector=nixl_pull/native_serving also match GLM0.28.0 and are not correct substitutes. Both Step adapters share the capability, not object identity;0.10.2/0.28.0 stay false. P12 adds adapter_record to the existing import while retaining RUNTIME_ADAPTERS for its CLI choices at:400. A future derived member needs one registry identity/capability entry, not caller family switches.

Scope arithmetic (inclusive added/deleted estimates): native178–265 + T1 140–210 + T2 240–390 = **558–865**; P1/P3–P13 = **427–689**; Dockerfile18–28 = total **1003–1582**. Additional test coverage grows existing P8/P9/T2 rather than silently adding a suite. P13 is counted separately; old `step5_dispatch_probe.py`, HTTP test module, native graph files, `r2_observation.py`, `build_stage.py` and `canonical_replay.py` remain unchanged. If implementation needs another file or exceeds the approved final bound, report before expanding, not promise an unrealistically narrower fix.

### Imported-model provenance, not just installed pins (P5/P6/P7/P9)

The existing `r2_runtime.SOURCE_MODULES` omits `vllm.models.step5.nvidia.model`; existing `Observation.verify_module` at `r2_observation.py:114–123` checks bytes and emits the path but does **not** reject a same-byte shadow copy outside installed root. Therefore:

1. P6 adds **model + forward_context + router_factory + routing_simulator_router** to SOURCE_MODULES. P5 already derives CANONICAL_SCHEMA_MODULES from this tuple; it hashes all four actual exported source files and declares the exact installed vllm root only for the new derived policy. It checks each digest against the corresponding existing installed-source listing in P7. Model now has both a schema hash and its pre-existing installed pin; the latter alone never proved import provenance.
2. At the existing post-load `bind_model_state`, inspect `sys.modules[name]` rather than import a second copy; require resolved `__file__` and `__spec__.origin` to equal the declared installed root's exact module-relative .py path. Find the actual loaded Step text module inside the current model/wrappers and require its class/forward source to be the checked model module/path; wrappers are not a reason to check only model_state. Reuse current observer.verify_module for actual hash reads; emit existing `provenance` events with origin status and the configured model/router/backend census fields, not a new event that current r2_verdict ignores/rejects. A byte-identical foreign copy must emit `verified=False` and raise an **observer** error before any positive root-qualified claim. No new import hook/global verification interface or verdict-event schema change is needed in r2_observation.py/r2_verdict.py.
3. Preserve the existing safe post-load observer boundary: mismatches/missing modules disable diagnostic evidence and mark UNVERIFIED without replacing native IO/exceptions/results. No routing guard reads tensors. Historical sealed packets stay unchanged; rebuilt legacy adapter retains old verification behavior, while strict origin/class/census checks require the explicitly declared derived policy/root, so legacy mocked host-runtime tests do not acquire unsupported derived requirements. The new policy cannot fall back to legacy behavior. Current verdict retains observer_error as invalid instrumentation; positive all-rank provenance remains a parent live acceptance check, not a fabricated G3 pass from static hashes.
4. P9 tests correct root/hash, same-byte foreign `__file__`, wrong `__spec__.origin`, wrong hash and missing actual model module with a real observer; also verify native call count/result/exception semantics. P8/P7 reject missing/mismatched model schema hash or declared root. **Installed pins remain53**: model was already one of50, while only the three context/router paths are added.

**Not owned/countable here:** parent Frontier logical-position workload/count/profiler semantic and profile-input schema/gate changes. Parent adds its independently designed file/line totals once. Existing clean-canonical or profiling launchers have their own old-image locks; they cannot be declared ready by this D24 package scope and belong in parent's profile/acceptance design. Avoid rewriting all historical JSON profiles or frozen sources. Derived adapter uses same installed vLLM version but **different digest and source bytes**; version equality is not source identity.

## 4. Verification strategy: CPU versus real GPU

### CPU before image build

- Compile new native files and source-method tests with Python3.12; exact E352/K8 positions0,43,44,6128,6144 and supplied shape/dtype controls. IDs distinct within each unmasked row; conserved8 routes per valid row, zero per masked row. Contiguous44 tokens cover each expert once; repeated-position requests intentionally repeat experts.
- Compare split[0,6128)+[6128,6144) and chunks[2048,2048,2032,16] with independently evaluated full[0,6144) IDs; q1 positions6144..6147 begin224/232/240/248. Padding with stale positions does not change valid IDs; masked rows=-1/0. Do not assert arbitrary multi-request waves max-min<=1.
- Weight controls use **same supplied logits** and selected unbiased sigmoid values, normalize on/off and non-default supplied scale, not weight=1. Preserve old random/normal outputs given their existing seeded controls and ordinary selector-empty factory. Missing positions/device/shape fail with explicit errors; no fallback/readback.
- Actual Step outer-forward CPU seam checks tensor **identity**, scoped cleanup/nested invocation and preservation of the same ForwardContext/MoE counter. Source/actual config controls establish default breakable mode and no compile decorator. CPU contract probes alone do not prove CUDA graphs or numerical G3.
- P8 runs actual `r2_replay.run` from a staged engine file for old and derived Step adapters; spy on version received by local_model_config/build_app/replay, including PD and direct-D reference, and reject a conflicting definition/config/engine.json identity before HTTP. Reuse its existing actual proxy fixture and `test_step5_http_capture.exchange` for derived fresh/reused-EOF evidence; registry false-capability controls keep non-Step receipts unchanged. P13 supplies the required engine-config argument in the existing reference fixture; no fallback or old probe rewrite.
- P5/P6/P7/P9 check model import origin/root/hash, including byte-identical shadow files; installed model pin agreement alone is insufficient. Reuse existing staged collector tests (D24 raw/mean, pressure actual pre-grant demand, HTTP, startup), source/hash/origin verifier, bash/compile/node preflight and offline ResourceConfig/dry-run. Never instantiate RJobBackend as part of offline validation. No CPU-overhead model/sampling suite.

### Parent-owned bounded GPU checks in derived image

T2 has no submitter/backend/auth code. Run it as a bounded test in parent's approved worker, not local GPU. Import actual installed derived native modules from neutral cwd, not source shadowing. Capture **all-padding dummy** first, then update the same device positions/mask buffer and replay with real q1/tail rows: this catches the particularly dangerous false success where capture-specialized all-padding routing never becomes live.

- CUDA eager; **actual BreakableCUDAGraphWrapper** PIECEWISE with current descriptor and a genuine test-only eager break as specified below; FULL `torch.cuda.CUDAGraph` capture under context mode NONE, matching native FULL capture. Two replays update positions/mask in place, with no outer Step Python setter/model/router call on replay. PW's recorded **eager callback does execute**; its counter must advance, while captured routing afterward uses stable tensor addresses. Match IDs/weights/output identity to independent expectation. Nested context/failure restoration runs outside capture. Bounded test output checks can synchronize after execution; production routing gains no readback/collective.
- CPU platform direct-MoE seam is insufficient: exercise the existing opaque `torch.ops.vllm.moe_forward` route through a thin real Step forward fixture for both selected modular methods, with actual router and deterministic output sink. A tiny fixture proves binding/capture semantics, not full expert math.
- T2 explicitly reuses old dispatch probe's **components**, not its entrypoint: `create_native_config`, `load_native_layer`, `DispatchObserver.collectives/experts`, `native_backend` and native distributed/workspace setup. The old `run_probe:454–456` calls Step5MoE under plain ForwardContext with neither positions nor mask; it **cannot** qualify the new mode and stays unchanged. The separate context/mask-bound module checks below keep `moe_backend=auto`, frozen TP8/DP2/EP16 and E8M0 predicate; record actual expert classes rather than forcing an oracle choice. CPU source checks/thin graph fixture do not prove all16-rank module dispatch or full serving.
- Full native package: P6128+D16 versus independent Dfull6144 reference, chunk/boundary/q1, real+dummy/padded lanes, allTP agreement, FULL runtime use, bit-exact transferred prefix and current independent-state criterion, no copied actual top-k. The existing D24 diagnostic adds state values and D32000 pressure evidence; missing numerical state agreement stays UNVERIFIED. All>=32GPU auxiliary checks are soft gates; native IO/training failure handling unchanged.

#### T2 genuine eager-break replay contract (review P2-3)

Use native `vllm.compilation.breakable_cudagraph.eager_break_during_capture` in the **test-only** fixture, not a fake capture counter. Its documented in-place rule (`:67–118`) and `add_eager`/replay implementation (`:198–217`) require a preallocated stable output, not a newly allocated return tensor.

1. Thin text-model fixture enters the actual derived `Step5ForCausalLM.forward` binding. Inside its model sink, run a small captured operation into a stable tensor, invoke the decorated eager callback which writes another preallocated tensor in place, then execute the actual derived router **after the break** via the existing opaque MoE seam. Test sink/router output buffers are also stable. Make a downstream output depend on the callback's updated tensor so a skipped callback cannot pass by coincidence; route IDs/weights independently prove positions/mask are current. No expert/backend replacement is used as full-module evidence.
2. Configure the existing PW runtime without toggling graph files/backend. Capture all-padding buffers, then inspect the actual wrapper entry (`entries[descriptor].capture`): require `num_eager_breaks > 0` and `num_graphs >= 2`. A single-segment graph fails this test even if direct router math succeeds. Retain counter/address evidence, not merely wrapper class name.
3. Change the **same** device positions, mask and callback-input buffers to real q1/tail rows and replay twice with different data. Assert callback execution count advances once per break per replay, unchanged addresses for callback/routing/output tensors, updated dependent output and correct post-break route IDs/weights. Assert outer Step/model/router Python capture counts do **not** advance on replay; scope cleanup remains correct outside capture. Callback must not call the new router or read `routing_positions` through replay-time Python context, which would test a different architecture.
4. FULL is a separate capture of the native outer-forward route under context mode NONE with no BreakableCUDAGraphCapture active; it must not be mislabeled PW/eager-break coverage. Real native attention/connector breaks in full serving remain separate live evidence, not proven by this bounded fixture.

#### T2 actual modular native-module evidence (review P2-4)

Build the native config from the exact supplied Step metadata with **existing** `create_native_config(args):328–345`. That function intentionally sets `enforce_eager=True` for module observation; record this fact and do not claim it reproduces serving PW/FULL. It retains text/quant mapping, auto MoE backend and native collectives. PW/FULL binding uses the separate actual-runtime config fixture above, not this eager helper's enforce_eager flag or a silent backend flip.

- Use actual native `load_native_layer(config, case, device):348–361` for FP8 layer3 and BF16 layer88; preserve dimensions/quant exclusions/dummy loading exactly. Reuse existing `LAYERS`, `DispatchObserver` and `native_backend:364–376`; initialize groups/workspace through the existing vLLM functions on parent-provided ranks. Import component functions without invoking old `main`/`run_probe` or changing its files.
- Preallocate **per-DP local physical-row** device position/int tensor and boolean mask from explicit logical-position test intervals. Supply `num_tokens=physical_rows`, known CPU physical sizes across DP and `is_padding=mask` to native `set_forward_context`; inside it enter actual `set_routing_positions(positions)` before `layer(inputs)`. No added routing collective/readback, no CPU mirror-derived offset. Use source6128+dummy1, tail16+dummy and padded-q1 cases only within the supplied approved worker budget; DPlocal logical positions need no wave reset. Masks exclude dummy/padded rows while existing collectives retain physical-row schemas.
- Wrap real `layer.experts.routed_experts.quant_method.moe_kernel.fused_experts` with observer.experts and actual GroupCoordinator with observer.collectives around the eager module calls; do not substitute a deterministic sink here. Record configured `position_cyclic`/Simulated router, source/model origins, quant method/oracle/backend/E8M0 and **actual** executed experts classes. Bounded test-only route capture validates IDs/weights/mask after execution; no numerical tolerance or copied actual reference top-k.
- Existing observer/schema-only errors remain soft, and native execution/IO errors preserve their failure semantics. This establishes actual modular dispatch under the new context/mask, **not** original-model semantics, G3 numeric agreement, production graph qualification or op-gap timing. All of those remain UNVERIFIED until their own live checks.

T2 expansion does not modify `step5_dispatch_probe.py` or native `breakable_cudagraph.py`; it legitimately reuses tested components with the new required inputs. If their interface cannot support the actual frozen path, stop and report additional files/lines rather than run the unbound old entrypoint or a fake backend as evidence.

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
5. Before GPU submission, parent exports actual installed **derived** native sources and fresh `installed_source_pins.txt` from that image. Only `models/step5/nvidia/model.py` among the four edits already occurs in50; add the three context/router paths, giving **53 installed pins**. Separately P5/P6 bind actual imported **model plus context/router modules** to exact installed root/origins and schema hashes; updating the existing model pin is not evidence that serving imported it. Reject even a byte-identical foreign-root model. Verify count from actual allowlist, not blind50. Reuse existing pin/hash parsing; nested manifest hashes or unchanged package version do not prove import identity.
6. After code integration and image digest are final, use the **existing canonical G3 builder** with new adapter/pins and new unused generation (capture04 if unused; otherwise next unused), same P/D workload/topology/ports/API/counts/context/128seq/runtime/criteria, P automatic pool, D32000 including independent reference, H80032/codesign nonpreemptible,90min/48GPUh. D24 collector raw/mean and original pressure/ownership/READ checks remain. Cloud path is an approved personal `/mnt/codesign-exp/ycfeng/...` string until parent operates it. Capture03 stays sealed/unsubmitted/not balanced.
7. Exact effective env for both roles must include selector `position_cyclic`; staged engine.json/config engine/definition declare the same derived adapter and R2 model/proxy/replay callers consume it. Both Step adapters enable registry-declared HTTP capture, not dictionary object identity. Validate image identity on node preflight, actual model/context/router imported origins on all ranks, source/config/tool hashes after integration, ResourceConfig/dry-run offline and fresh creator/mount/GPU/live evidence by parent. Keep launcher alive during any later platform job. No submitted/job marker is created by this design.

Thus 53 total is the intended minimum, not 50 unchanged or 54. Other inspected read-only route/oracle/runtime sources remain provenance witnesses and can be retained in derived-source receipts without pretending they were code changes.

## Design validation / unresolved boundaries

Observed in the first design: actual locked configuration enables breakable Step PW/disables torch.compile; FULL capture context mode NONE; both modular routes accept precomputed top-k; read-only capture03 manifest/pin checks passed. Its recorded manifest remains `cdaad24ecded6da6f2eea90ed4ef4a582432db2fd75c56c29aafb22f2f56a57d`; those checks were not rerun in this correction and are not new live evidence. This correction reread only the actual caller/adapter, source-verification, native breakable graph and existing dispatch/test seams needed by review. Document-only sum/path-uniqueness and Git diff checks are the local validation; **no native/CPU routing probe, runtime test, image/GPU/platform command or production edit** ran. Parent Frontier/profile design was not investigated. Proposed graph/model/weight/backend/G3 checks remain UNVERIFIED.

Native design is ready for the parent's **combined numerical scope approval**. Remaining live questions (graph replay pointer dependence, actual auto oracle/E8M0 subkernel, exact selected-score CUDA behavior, full G3 state agreement and actual new profile/op-gap timing) are verification items, not reasons for speculative additional code or immediate wider research. Proceed workflow -> GPU/op-gap -> CPU overhead last.

### Precise response to independent review

Reviewer task: `ses_e9e15d18cffemTkCOdK4P941nU`; findings were supplied by the parent. This document correction addresses the design, not completed implementation or reviewer acceptance.

| P2 finding | Corrected design / decisive inspected seam | Counted scope and regression | Validation status |
| --- | --- | --- | --- |
| Missing clients / Step HTTP adapter identity | Actual `calibration/pd_proxy.py:203`; registry has no HTTP capability. P3 declares it once; P12 consumes it. P11 reads required actual engine.json and propagates selected version through all3 callers, including direct-D. Existing worker already passes engine-config. | Add P11/P12; expand existing P8 HTTP/R2 regression. Count unavoidable P13 required fixture argument; no fallback. | Design source inspection PASS; implementation/client tests NOT_RUN. |
| Installed model pin mistaken for actual provenance | Existing SOURCE_MODULES omits model; verify_module only compares bytes. P5/P6 add model schema/module plus exact installed-root/spec/class/hash binding; P7 checks stage correspondence. | Expand P5/P6/P7/P9; same-byte wrong-root/spec test in existing P9. Installed pins remain53, not54. | Design seam PASS; actual all-rank import evidence UNVERIFIED. |
| PW single segment lacks actual eager-break coverage | Native decorator + actual wrapper entry counters, stable in-place callback output, router after break, dependent output and updated positions/mask over2 replays. | Expanded existing T2; num_eager_breaks>0/num_graphs>=2 mandatory, no native graph-file edit. | Test strategy specified; actual graph replay UNVERIFIED. |
| Old native probe omits positions/mask | Old run_probe not reused. Reuse create_native_config/load_native_layer/DispatchObserver/native_backend with explicitly bound real new context and mask; distinguish helper's eager module evidence from PW/FULL. | Expanded existing T2; old probe unchanged, no unbound old CLI evidence or backend flip. | Test strategy specified; actual modular dispatch/G3 UNVERIFIED. |

Ready for parent numerical-scope proposal: **19 unique files /1003–1582 added+deleted lines** for this native-facing correction only. This design document and generated source/image receipts are not code/test-file count substitutes; parent unions its independent Frontier/profile paths before requesting approval.

Local document checks printed native source4/178–265, native+tests6/558–865, package code+tests12/427–689, Dockerfile1/18–28 and total19/1003–1582 (**PASS**); `git diff --check` returned no whitespace errors (**PASS**). This checks document accounting only; it is not a native routing/CPU probe, runtime test or acceptance gate. Reproducible command (owned worktree; no output files):

```bash
env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B - <<'PY'
from pathlib import Path
import re
text = Path('task_memory/task_2026-10-22_step5_d29_native_balanced/native_design.md').read_text()
pattern = r'^\| ([NTP]\d+) \| (?:F )?`([^`]+)`.*?\| (\d+)–(\d+) \|'
rows = [(key, path, int(lo), int(hi)) for key, path, lo, hi in re.findall(pattern, text, re.M)]
assert len(rows) == len({row[1] for row in rows}) == 19
groups = [
    ('native_source', [r for r in rows if r[0].startswith('N')], 4, (178, 265)),
    ('native_and_tests', [r for r in rows if r[0][0] in 'NT'], 6, (558, 865)),
    ('package_code_tests', [r for r in rows if r[0].startswith('P') and r[0] != 'P2'], 12, (427, 689)),
    ('dockerfile', [r for r in rows if r[0] == 'P2'], 1, (18, 28)),
    ('total_unique', rows, 19, (1003, 1582)),
]
for name, subset, count, bounds in groups:
    actual = (sum(r[2] for r in subset), sum(r[3] for r in subset))
    assert len(subset) == count and actual == bounds, (name, actual)
    print(name, len(subset), actual, 'PASS')
PY
```

## 1. 规划完成情况

**汇总：** PASS 4｜BLOCKED/FAILED/SKIPPED 0｜TODO 0（仅本次四项设计纠正，不表示实现或 GPU验证通过）

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 实际 client adapter／HTTP capability | PASS | P3/P11/P12/P13 与既有 P8 regression，真实路径已核实。 |
| imported model origin／hash／root | PASS | P5/P6/P7/P9，明确拒绝同字节 foreign-root copy；53 pins不冒充 import evidence。 |
| 真正 eager-break graph 检查 | PASS | T2 要求稳定输出、实际 counters 和 break后更新路由；运行待验证。 |
| 正确 native module复用与范围 | PASS | T2 显式 positions/mask，新预算19 files／1003–1582行，不改旧 probe。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 四个 P2 review follow-up 至今｜**当前前沿：** 提交修订本文并交回主会话合并范围。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 只补 review明确缺失的 caller/provenance/T2策略，未执行或扩展 harness。 |
| D2 细节反复重试 | CLEAR | 无 production修复或重试，只进行一次设计纠正。 |
| D3 支线问题修补 | CLEAR | 检查只服务四项 P2，不调查其他 owner的 Frontier/profile。 |
| D4 可并行任务停滞 | CLEAR | 用户明确本子任务不再 delegation；父端独立设计未被阻塞。 |
| D5 超范围镀金 | CLEAR | 仅此 design 文件，无 production/native 改动。 |
| D6 分析空转 | CLEAR | design 即本次交付，调查形成文件表与决定。 |
| D7 路线违背 | CLEAR | 未执行 image/GPU/platform 或 CPU-overhead 工作。 |

**文档检查：** 当前主工作树 authority／plan／progress已在9ed5763区分历史和当前 scope。S2/S5：旧 native_design缺 client与实际 model-root检查，并把单段 PW／无binding旧 probe当作可复用验证；已在唯一 owned文件内纠正、重算预算。主会话 plan:20–23/progress:15–19记录的是首版估计，完成本次集成时须更新为本表新总数。S1／S3／S4／S6 未发现明显缺陷。
**归因：** 无执行偏离。

## 3. 纠正建议

**分支：** B：设计缺口已在本文修订，集成后更新主端估计再申请统一范围。

**文档更新（先执行）：**
1. `native_design.md`：补齐四个P2、唯一文件表和1003–1582行预算。（已执行；仅设计，未改变用户 policy。）
2. 主端 D29 `plan.md:20–23`／`progress.md:15–19`：集成修订后记录新估计与 review remediation状态。（可直接执行，父端所有权，本子任务不编辑。）
3. 同一 semantic correction的最终数值授权：主端合并独立 Frontier/profile范围后一次询问。（需用户确认，本文不授权实施。）

**需停止的偏离行为：** 无。

**接下去的路线：**
1. 本子任务提交修订本文 → 完成标志：仅本文 commit，branch clean。
2. 主会话并行汇总 Frontier／profile 设计 → 完成标志：同一 semantic correction 的完整文件／行数范围。
3. 一次范围批准后再实施 → 完成标志：reviewed source 与 CPU／GPU／image／new-generation 证据；不得提前宣称 G3 成功。

## 4. 最值得关注的 3 点

1. 新完整估计为19 unique files／1003–1582行，含实际 callers及不可省略的现有fixture参数更新。
2. PW必须有真实 eager break；model必须检查真实 import root；旧 unbound probe不是新证据。
3. 所有实现／GPU/G3项目仍 UNVERIFIED，保持 workflow → GPU/op-gap → CPU overhead；原 capture03不重标。
