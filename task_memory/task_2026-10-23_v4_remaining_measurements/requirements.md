# V4 kernel-only attention costs

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Allocated the pre-existing32-rank positive-fixture repair as independent queue-time CPU work; frozen checkers/runtime/sealed packages stay unchanged. |
| 2026-10-24 | Actual-image native autostop/export controls passed; allocated fresh profile04 with only the existing optional CUDA-table field disabled. |
| 2026-10-24 | Selected the existing CUDA-table toggle for bounded CPU phase controls; no timeout, capture-window or runtime change is selected. |
| 2026-10-24 | Opened bounded offline diagnosis after profile03's launcher exit1; immutable failed-run artifacts and the original attempt remain preserved. |
| 2026-10-24 | YC approved the fresh callback payload-correction probe and standing authority for similar bounded diagnostic repairs/probes. |
| 2026-10-24 | Allocated two existing non-core files for source-backed coverage and shutdown corrections after CPU lifecycle reproduction. |
| 2026-10-24 | Root allocated optional diagnostic DP sequence recording after exact-image WS2 proof, including direct dummy-execution annotations while preserving frozen base inventory schemas. |
| 2026-10-24 | Root allocated the reproduced collector CUPTI-loader repair and a targeted single-H800 stream/capture probe under the existing CPU/H800 authority. |
| 2026-10-23 | Recorded YC's approval of the recommended kernel-only route and CPU implementation/measurement-design scope. |
| 2026-10-23 | Root allocated the two reproduced native collector launch seams under existing repair-scope delegation; GPU authority is unchanged. |
| 2026-10-23 | YC requested completion/commit/handoff of current work; root allocated bounded repairs for five unique collector review findings. |
| 2026-10-23 | Recorded completed H source/CPU integration64f3c950 and preserved G2 as the next-session scope. |
| 2026-10-23 | YC resumed G2 in the current session; existing CPU implementation/design authority and separate GPU approval remain in force. |
| 2026-10-23 | Recorded the seven-file G2 CPU allocation and independently reviewed failure-isolation repairs within the unchanged750-line budget. |
| 2026-10-23 | Accepted G2 CPU source 05952396 after both re-reviews, final separate-suite verification and explicit matrix binding; new GPU authority remains pending. |
| 2026-10-23 | YC approved the native profile run and granted standing authority for task-required H800 RJobs of any necessary scale. |
| 2026-10-23 | Recorded the owner-scoped staged-entry repair and fresh profile02 generation after the independently reproduced profile01 import failure. |
| 2026-10-23 | YC retained the queued32-GPU run and prioritized independent tickets, including useful8-GPU or smaller pending work. |
| 2026-10-23 | Root selected the source-backed four-point local-MoE supplement and allocated a two-file non-core package under the existing queue-time/H800 authority. |

## Original requests and decision

[Original Request — execution order] “routing暂时统一采用balanced distribution方案对齐（frontier和vllm的routing都采用均衡分布的设定）”; “先进行workflow和gpu侧的对齐（包括op gap计算），最后再进行cpu overhead（gap）的计算测量”.

[Original Request — proposed choice] “后续 V4 注意力总耗时采用哪条路线？这次选择确定 CPU 实现和测量设计；新增 GPU 运行会在具体方案就绪后单独申请。” Recommended: “以 KERNEL_ONLY 为算子成本，依据原生依赖处理 GPU 流重叠。先补齐覆盖和归因证据，不直接累加现有 scope，也不把未知差额归为 CPU gap。”

[Original Request — YC response] “批准，继续”.

[Original Request — session close] “请你完整当前正在执行任务的收尾，然后完整提交更新docs和codes，生成一段handoff prompts，我将在其他session中继续执行”. Close the active collector source/CPU findings, verify and consolidate accepted code and records on the main branch, and produce a self-contained continuation prompt. Future driver/runtime/GPU work belongs in the handoff.

[Original Request — resume here] “暂不执行handoff，继续在当前session推进任务”. Continue G2 in this session from committed77668066. This supersedes the scheduling instruction to wait for another session; the completed H acceptance and the requirement for a separate concrete GPU approval remain unchanged.

[Original Request — H800 proposal] “是否批准一次 32×H800 / codesign、非抢占式诊断运行？保持 P16+D16、TP8/DP2/EP16；整次运行最多 60 分钟、32 GPU·h，在首轮 formal 采集每个 worker 的前 64 个 steps。先完成新包与锁定镜像检查，再提交；此申请不包含自动重跑。”

[Original Request — YC H800 response] “批准；h800集群的的运行无需向我过问，允许你提交符合任务需求的任意规模的rjob（在h800上）”.

[Decision D-V4-H800-RJOB-AUTHORITY — 2026-10-23 UTC] The current 32-H800 native profiling run is approved. Task-required H800 RJobs no longer need repeated resource approval, including necessary changes in scale. Respect actual quota/capacity, personal StepMind submission, local source mounts, storage boundaries and auxiliary soft gates. This resource authority does not change calibration predicates, native/C26 source approval or unrelated paused task scope. The first run retains the recorded 32-H800, 3600-second, 64-worker-step plan; any subsequent run records its concrete purpose, resources and evidence before submission.

[Original Request — queue-time priorities] “继续等待32卡的gpus资源，优先进行其他tickets等todo事项（可以先进行8卡等规模的待处理任务）”.

[Decision D-V4-QUEUE-PARALLEL-WORK — 2026-10-23 UTC] Keep profile02 queued and its launcher alive. Prioritize independent unfinished DSV4 ticket work; smaller H800 measurements may run under the existing standing authority once their missing tuples, source/configuration, resource ceiling and acceptance are recorded. A smaller topology qualifies only its measured scope. Existing GLM pause, native/C26 boundaries and CPU-overhead-last order remain in force.

[Decision D-V4-KERNEL-ONLY-GPU-COST] Adopt the recommended per-operator KERNEL_ONLY route. Implement V4-private input/consumer prerequisites and design measurements from native source and retained evidence. Kernel-to-scope uniqueness is not completeness or wall-clock additivity. Serving attention totals require complete attribution plus explicit GPU-stream/dependency accounting. CUDA_EVENT stays separate diagnostic/equivalence evidence; the unexplained uncompressed difference is not labelled CPU overhead or removed by scaling.

## Scope and acceptance

- Continue parent05/09 under approved D-Q10 measured consumption. Preserve balanced_topk and all layout/source/routing/phase/graph/provenance refusals.
- Reuse the selected attention-file path and private sidecar/admission pattern. No new shared config field or registry mechanism is needed for input bundles. Inspect actual callsites before choosing the integration point.
- Audit the accepted prefill trace and pinned native callgraph first: measured/correlated kernels, missing/ambiguous ownership, within-stream order, cross-stream dependencies and unsupported evidence. Keep all samples and original bytes.
- Produce a bounded design before core edits. Default domain limit: five frontier/ files/500changed lines, one owner per file; non-core audit/tools/tests separately accounted. Adjustments follow YC's existing root scope delegation; new shared interfaces/calibration rules remain separate decisions.
- A private input component may be accepted independently, but isolated TP1/eager/six-layer data never become full serving/TP8/DP2/EP16/graph evidence. Missing evidence yields a typed error, not a cost fallback.
- Verify real accepted inputs/metadata, all94 isolated medians, broken provenance/layout/coverage controls, existing affected tests, independent review and the required default matrix against a fixed source. Non-simulator changes may use explicitly bound existing matrix evidence.
- Measurement preparation targets actual per-role TP8/DP2/EP16/PP1/native V2. Existing DSA/V1 replay, EP16-only or TP8-only runs cannot establish equivalence. Inspect the existing native worker/extension path before proposing shared profiling parameters.
- D-V4-KERNEL-ONLY-GPU-COST originally authorized CPU implementation/design only. New H800 runs now use D-V4-H800-RJOB-AUTHORITY, rather than reusing the spent prefill/native06 allocations. Publication, native/C26 edits, unrelated broad CUPTI investigation and CPU-overhead collection are not granted by these resource decisions.

## Current reference

### H launch integration allocation

Root authorizes the delivering H owner to add `tests/comparison/dp_placement_pp/vllm_replay.py` (8 changed lines) and `tests/comparison/calibration/pd_replay.py` (4 changed lines). `collector/launch_seam_receipt.json` at owner commit `9366b6e4` reproduces all three blockers: the new extension is rejected for balanced routing, its worker PYTHONPATH is absent, and its profiler configuration is discarded. Reuse one extension registry across these launch paths and `native_pd_worker.py`; no parallel mechanism or permissive fallback. H's allocation is now five non-core files / 787 changed lines (450 collector, 25 native worker, 300 tests, 8 command builder, 4 P/D environment). Existing per-step local/synchronized dispatch evidence remains required. This is necessary wiring for the approved collector, not new calibration semantics or a GPU run approval.

### H review repair allocation

The preceding787-line allocation is superseded for necessary review corrections only. At d94b821a, independent reviews found one P1 and four unique P2 issues: graph-hook lifecycle, slot-mapping ownership, actual source-visible PDL arguments, atomic step-retention admission and bounded detailed error logging. Root authorizes the same five-file owner to use collector550/test400/native-worker25/command-builder8/P-D-environment4, total987changed lines. Record exact changes and RED/GREEN controls; do not remove controls or compress implementation to fit. Both review reports are retained in `collector_primary/review_2026-10-23_initial.md`. No new shared surface, native/C26 change, driver implementation or GPU authority is granted by this repair allocation.

Native06 is ACCEPTED_NATIVE_REFERENCE_R4: originalb62c155, reader integrationd278659, bound by parent `native06_acceptance_20261023/{primary_acceptance_r4.json,frozen_r4_result.json}`. Isolated consumer integratede89b45d:144unit/94exact/legacy102-72, fixed576d278 strict77. Original R3 FAIL/native05 UNVERIFIED remain historical. GLM remains paused; four-metric calibration is unfinished.

Session-close outcome: the private input component is accepted at732406aa; H sourcef8621a8 is accepted and integrated64f3c950 within880/987lines, with five findings closed and merged verification complete (`collector_primary/primary_acceptance.json`). G2 driver/window implementation and a concrete future GPU request remain in `progress.md` and `handoff_prompt.md`; this outcome grants no additional run authority.

### G2 implementation allocation and current authority

Under the existing CPU implementation/design approval, root owns the actual replay driver, window retention and CPU verification after the two initial design delegation attempts failed. The final allocation is seven existing non-core files/750changed lines (production350,tests400), with current owner-checked change433lines. Paths and lifecycle are in `driver_design.md`; `driver/cold_start_scope.md` records the additional existing test path and its reproduced baseline mismatch. No core/native/C26 interface or calibration rule is changed.

Both initial independent reviews reported three distinct failure-isolation defects. The delivering root owner reproduced and repaired all three; both R1 reviews closed them with no new findings. Source `05952396` is now `ACCEPTED_CPU_PROFILE_DRIVER_COMPONENT`, bound by `driver/primary_acceptance.json`, final separate-suite results and an explicit existing 77-case matrix binding. The first-formal 64-step/32-H800 resource sheet in `proposal.md` is approved under D-V4-H800-RJOB-AUTHORITY. Package/source/image checks and runtime qualification remain required.

## Staged-entry correction — execution decision

Independent profile-package review found that flattening `native_pd_worker.py` to the package root invalidates its repository-relative `vllm_replay` import. Root reproduced the exact `ModuleNotFoundError` in a fresh process. Under the existing repair authority, the original package owner corrects only the generated launcher/bootstrap import paths through the existing builder and profile plan, within the same two-file/300-line allocation. Accepted runtime bytes remain pinned to `05952396`. Preserve the unsealed profile01 package and failed evidence; generate fresh `candidate_h800_p0_profile_02` with tag `dsv4-h800-p0-profile-20261023-02` and the same approved resources/window. Both entry paths require fresh-process verification without preloading `pd_replay` or supplementing the dependency path outside the generated entry.

## Q2b local-MoE package allocation

Under D-V4-QUEUE-PARALLEL-WORK and D-V4-H800-RJOB-AUTHORITY, root selects EP16 rank0 gathered populations16/32/6152/12288 from corrected `moe_grid_design.md`. The6152 case includes a dummy peer and is conditional evidence. Reuse the existing producer with physicalTP1, balanced distribution, `routing_runtime_path=balanced_topk`, and separate CUDA_EVENT/record_function invocations. Retain all columns; only the grouped-GEMM envelope is the candidate cost scope. Gate rows are not native-local rows, and shuffle overlaps that envelope. No rank/time/weight/graph/serving equivalence follows from this design.

One package owner may add at most `moe/build_package.py` and `moe/profile_plan.json`, total300 gross changed lines outside core; generated immutable inputs and records are separately counted. Worktree `/data/ycfeng/Frontier/.worktrees/v4-moe-local-run`, base56e9405d. Reuse existing worker/launcher/seal mechanisms without editing their retained originals. Target1H800/codesign/non-preemptible,16CPU,mem_gb128,1800s ceiling, all durable outputs under `/mnt/codesign-exp/ycfeng`. Root owns actual image/SDK gates, sealing, quota and submission. No core/native/C26/consumer/admission edits are allocated.

## Post-profile02 stream identity allocation

Profile02's accepted audit reproduced192 missing stream mappings. Root reproduced the exact unversioned-library failure in the locked image and verified that Torch already loads `libcupti.so.13`; its versioned SONAME exposes `cuptiGetStreamIdEx` without a path override or GPU initialization. The bounded repair may change only `tests/comparison/calibration/native_serve_profile.py` and its existing unit tests, at most80gross lines, deriving the ABI major from the running Torch CUDA build. Preserve the existing soft-error/raw-handle behavior; add no loader fallback, new environment flag or symlink.

After CPU verification/source review, prepare a targeted1H800/codesign/non-preemptible probe,16CPU/mem_gb64/600s ceiling, through the existing StepMind launcher/package mechanisms. Its purpose is actual collector stream-ID versus trace correlation and, independently, whether native profiler startup/formal cycles expose usable captured-node ownership. A new `tests/comparison/calibration/native_profile_probe.py` may use at most250lines; immutable inputs and task records are separate. Keep all small outputs and failure evidence, with exact image/source/library hashes. No model/serving cost claim, native/C26 edit, custom CUPTI callback subscription, numeric graph-ID remapping or revival of the paused GLM probe is authorized by this allocation. The old H200 graph probe's source/exec namespace mismatch and CUDA13.1-only tools-ID API must inform the new probe design.

The first249-line probe review found two P2 acceptance defects: graph correspondence did not require successful capture/replay completion, and graph parsing could suppress an already valid eager verdict. Root allocates the same one-file owner at most275lines for these repairs without compression, plus retained temporary CPU failure controls. Actual image imports/config/wrapper construction already passed on the249-line source; final source/package binding remains required after repair. This changes neither GPU resources nor the excluded APIs/scope.

## Diagnostic DP sequence allocation

The exact-image CPU control in `profile01/dp_sequence_cpu.md` establishes getter availability and symmetric counters on a measured two-rank gloo group. It does not establish native worker pairing. Frozen R3 requires exactly three `dispatch_evidence` keys; frozen R2 also fixes the hook set. The existing post-all-reduce observation is the sole native hook for this addition.

Root authorizes one owner to change four existing non-core files: `tests/comparison/calibration/{native_full_structure.py,native_serve_profile.py}` (40gross lines combined) and `tests/unit/{test_native_v2_serving_inventory.py,test_native_serve_profile.py}` (140combined), total180. Record raw counter state beside the existing inventory thread-local dispatch dictionary and emit optional `steps[].dp_post_all_reduce_sequence` before final detached-row capacity admission. Do not add it to retained base inventory rows, their dedup key or `dispatch_evidence`; no schema version, callback, hook entry, flag, collective or checker edit is needed.

Read the actual explicit `group=` operand; do not infer WORLD or silently accept missing/throwing getters. Reuse existing soft observation errors, preserving native return/exception behavior and the separately observed CPU vector. Update only local stand-ins to pass realistic groups, including the local namespace around the frozen oracle; never edit its retained source. Reset per-execution state so no stale counter reaches a later step. No duplicate-collective branch is needed beyond existing ambiguity handling unless a reachable case is demonstrated.

CPU acceptance includes nonzero startup counter, repeated/zero vectors, post-native-call ordering, inactive dispatch, missing/raising getter isolation, unchanged native failures, reset/detached-row/capacity behavior and frozen-checker positive/negative regressions. Bind the existing unchanged-domain77-case matrix explicitly. Future pairing compares raw values within `(run generation, role, ordered DP group members)` with lane/rank as member identity, without offset correction; it never equates counters to step indices. This is optional diagnostic evidence, not a new calibration predicate or native-reference admission.

The actual pinned `gpu_worker.py:1210–1212` also calls `_dummy_run` directly through `execute_dummy_batch`, outside `Worker.execute_model`. Profile02 rank0 snapshots retain6046/1647/203/12 dummy annotations with `step=null` on P0/P1/D0/D1. Therefore the same optional raw counter must also travel through the existing annotation/context envelope during active native dispatch. Do not invent worker steps for those calls. Post-execution/sample-token context uses its detached step's counter rather than a newer unrelated thread-local value. This placement correction remains within the same four-file/180-line allocation and requires direct-dummy and stale-context controls.

## Post-profile02 coverage and lifecycle allocation

Under the existing CPU implementation/repair authority, root selects `profile01/profile03_design.md` and assigns one owner two existing non-core files: `tests/comparison/calibration/native_pd_worker.py` (80gross lines) and `tests/unit/test_native_pd_worker.py` (180gross), total260. Add task-local `role_profiler_max_iterations` handling through existing per-node profiler plan construction, preserving input objects and per-role/lane paths. Package values512/64 and records600000/bytes268435456 remain a later explicit package change, not edits to sealed inputs.

Use known-peer drain paths for both the head quorum and retained membership, and the existing `peer_wait_seconds` for the headless post-stop-marker wait. Preserve post-stop-marker meaning, native stop escalation, native return/failure handling and the soft missing-marker observation. Retain narrow monotonic wait/stop timing in a separate diagnostic result field without changing frozen `shutdown_order` semantics. A finite wait is not a completion guarantee and local clocks do not prove cross-host order.

Acceptance requires source-preserving CPU RED/GREEN controls for the delayed head marker, visible direct paths with delayed enumeration, incomplete known-peer quorum, late/missing marker soft failure and ordinary native shutdown paths; verify role-specific profiler values and immutable inputs. Reuse existing tests, then independent reviews and root verification before integration. No core/native/C26/checker edit, new callback subscription, CPU-overhead inference or GPU submission is part of this source slice.

## Callback payload correction and similar diagnostic authority

[Original Request — fresh callback proposal, 2026-10-24] “旧 callback 诊断的3/3次数已用尽。官方 sample 已确认旧探针缺少一层 resourceDescriptor 解引用。是否批准修正一个新副本（最多80行改动），先做 CPU 检查，再最多进行一次1×H800／600秒验证，失败即停？目标是直接验证 graph 节点对应关系。”

[Original Request — YC response] “批准，类似请求无需申请权限，皆授权”.

[Decision D-V4-BOUNDED-DIAGNOSTIC-AUTHORITY — 2026-10-24] Approve the fresh callback payload correction and its at-most-one1H800/600s diagnostic run after CPU verification. Similar bounded task-required diagnostic repairs/probes need no repeated approval. Before execution record the concrete scope, resource/attempt ceiling, evidence and acceptance. Existing H800 platform authority applies. This decision supersedes the earlier callback exclusion only for this scope; it does not change native/C26/calibration predicates or resume unrelated GLM work. The old3/3 failed-attempt history stays unchanged; no blind automatic retry is selected.

One owner may create a fresh probe copy under `profile01/graph_probe/`, with at most80gross probe changes relative to the frozen attempt3 `probe_diagnostic.py`; inherited bytes are recorded separately. Keep its existing event/DAG scope labels, callback set, in-callback query order and strict9-node/5-kernel/4-event correspondence predicate. Correct the envelope indirection following the exact official13.0.85 sample. Root additionally allocates at most160lines for CPU regression controls, which must exercise the actual callback with allocated ctypes envelopes and check soft rejection of missing descriptors. Reuse existing probe checks; do not build a parallel mapper or change any old package/test source. Root owns package/image/SDK validation, seal, submission and actual-data acceptance.

### D-V4-GRAPH-PROBE-PACKAGE — root implementation allocation

- Prepare a fresh immutable package for corrected probe1aa1e4a5 under the approved one1H800/600s experiment. Reuse the existing stream-probe launcher/ResourceConfig pattern; no parallel submission or automatic retry. Package/config maintenance allocation<=220gross lines, generated launcher<=100lines reported separately; this changes no probe/native source.
- Fixed resources: H800/codesign, non-preemptible,1GPU/1replica,16CPU/64mem_gb,600s; image2286e853 and existing CUDA13.0 compat path. Local NFS source, personal cloud mount and fresh personal destination follow the existing handbook. No credentials in artifacts.
- Retain diagnostic stage/fault sinks in a sibling directory created before CLI entry. The probe's --output-dir must remain absent until the probe creates it. CLI is `run --output-dir ... --image-digest ...`; FRONTIER_GRAPH_DIAGNOSTICS_DIR is required before argparse. Preserve close-before-Kineto logic and strict U1/U2/U3/U4 verdicts.
- Root owns actual-image/SDK checks, current auth/quota/mount/destination validation, final source review, seal, launch and raw-data acceptance. Preparation performs no platform operations and grants no runtime qualification.

### D-V4-PROFILE03-PACKAGE — root implementation allocation

- Build fresh native profile03 from integrated runtime1437ace6, accepted by `profile01/integration_cpu/primary_acceptance.json`. Reuse the existing `v4_p0_native_20261021/build_candidate.py`; only its profile branch and a fresh `profile01/profile03/profile_plan.json` are maintained changes,<=140gross lines combined. Preserve all old plans/packages and native06 generation behavior. New freeze/proposal/acceptance bindings are derived records, not replacements for old evidence.
- Select generation03, per-role max_iterations512/64, records600000/bytes268435456, one first-formal window and the existing balanced canonical workload. Use fresh run tag `dsv4-h800-p0-profile-20261024-03`, run_generation `v4-native-v028-h800-p0-profile-20261024-03`, cloud root `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18/dsv4_h800_p0_profile_20261024`. Keep other profile settings and native/C26/workload/source/binary pins unchanged.
- Preserve resources:4replicas×8H800/codesign,64CPU/512mem_gb per replica,non-preemptible,timeout3600s,startup/replay1800s,peer wait1200s,stop90s,TP8/DP2/PP1/localDP1,required RDMA/local NFS/personal cloud mounts. Preserve existing soft diagnostic errors; add no auxiliary hard gate.
- Preparation verifies all13 runtime module identities against1437ace6, manifest closure, four node plans, actual staged CLI/import paths and the existing77-case binding; package reviewers/root then verify the actual image and SDK before sealing/submitting. Runtime acceptance still requires >=3 useful steps per role/lane, no unintended drops, actual raw DP pairing/stream joins and correct post-stop evidence. Captured per-op ownership is independent and remains UNKNOWN. Resource authority is D-V4-H800-RJOB-AUTHORITY with bounded diagnostic continuation under D-V4-BOUNDED-DIAGNOSTIC-AUTHORITY.

### D-V4-PROFILE03-FAILURE — offline diagnosis allocation

- [Original Request — runtime notification, 2026-10-24] The previously submitted launcher reported `Exited with code 1`. Continue the approved run verification and required failure diagnosis under D-V4-H800-RJOB-AUTHORITY and D-V4-BOUNDED-DIAGNOSTIC-AUTHORITY.
- Preserve profile03's sealed45 inputs and complete retained output inventory:190 files/8208668144bytes, downloaded hash-identically. The completed submission remains attempt1/1; this allocation submits no additional GPU run and changes no runtime/native/C26/checker/calibration behavior.
- Independent scopes: immutable-artifact timeline (two Markdown files<=120lines),32-rank collector metadata/DP-counter audit (two Markdown files<=110lines, scratch adapter<=180lines), and a source-bound CPU RPC feedback loop (two Markdown files<=110lines, scratch code<=220lines). Owners write only their named artifacts under `profile01/profile03/failure/`; primary owns platform evidence, raw-trace inspection, synthesis and acceptance.
- The CPU loop must exercise real native deadline/response handling with mocked transport or elapsed work, reproduce the exact `RPC call to execute_model timed out.` symptom, and distinguish that mechanism from reproducing the original GPU workload or proving which profiler phase was slow. No hand-written replacement timeout algorithm or mocked C++ profiler parser counts as evidence.
- Verify submitted/completed/failed/not-submitted request counts separately. Failed-run traces may support bounded diagnostic findings; file existence, an audit PASS or a complete gzip does not grant run/reference/serving/cost admission. Record any later repair's concrete scope, budget, regression checks and run ceiling before implementation/execution.

### D-V4-PROFILE03-STATISTICS-CONTROL — source-bound CPU verification

- The retained P1 rank0 trace contains2434560 kernels/4278830304 uncompressed bytes, versus P0 rank0's356946/574658585; complete JSON/gzip parsing and P0 equality with the original streaming reader are verified. All eight P1 gzip files have mtimes07:01:09–13Z and tables07:06:26–29Z. These are observations, not a measurement of Python event parsing or proof of the original outstanding-reply schedule.
- Exact-image Torch2.13 source and pinned native wrapper show trace export before `key_averages().table()`; the existing native `torch_profiler_dump_cuda_time_total` field defaults true. The proposed minimum correction is false in a fresh profile plan, preserving raw traces,512/64 caps, record limits, workload, native/C26/runtime bytes and the300-second RPC default. It intentionally omits the optional per-rank text statistics; it does not promise every future export fits300seconds.
- One CPU-control owner may add a<=180-line scratch adapter reusing the already verified native AST feedback loop and actual `_stop`/table methods. Verify true-table RED versus false-table GREEN with the same export cost; export-too-slow must remain RED, and CPU-only automatic tables must remain explicit. Mock only elapsed work/transport/statistics input, not native branching or timeout logic. Own only `failure/{statistics_control.json,statistics_control.md,statistics_control_command.md}`; combined Markdown<=100lines. Primary owns independent actual-image controls and final selection. No runtime/native edit or GPU run is part of this control allocation.

### D-V4-PROFILE04-PACKAGE — minimum configuration correction

- [Decision — 2026-10-24] The actual locked-image native constructor/autostop exported both CPU control traces with eight identical payload annotations. Dump=true called statistics/lazy-event materialization once; false called neither and omitted only the text table. Receipt `/data/ycfeng/tmp/glm53_dsv4/profile03_statistics_image_g8m3pvpv/receipt.json` pins native wrapper/Torch bytes and confirms CUDA stayed uninitialized. Native virtual-duration toggle controls remain required before seal.
- One package owner may change only the existing `v4_p0_native_20261021/build_candidate.py` and fresh `profile01/profile04/profile_plan.json` (<=100gross lines combined), plus the existing `tests/unit/test_native_pd_worker.py` (<=60gross regression lines). Reuse all accepted generation03 mechanisms; update only the root-selected profile generation guard and fresh identity/path bindings, with `torch_profiler_dump_cuda_time_total=false`. Preserve generation06's non-profile branch, all old plans/packages,13 runtime modules,87 native source/20 binary pins, native/C26/workload/routing bytes,512/64 caps,600000/268435456 limits and the300-second RPC default. Generated immutable package files and provenance records are separately counted.
- Prepare `candidate_h800_p0_profile_04` with fresh20261024-04 tag/generation/cloud root. Target one4×8H800/codesign non-preemptible run,64CPU/512mem_gb per replica,3600-second ceiling, TP8/DP2/PP1/EP16, with the existing personal local-NFS/cloud mount/RDMA configuration. This is a correction-backed validation under D-V4-H800-RJOB-AUTHORITY, not a blind repeat. Root owns quota/resource choice, image/SDK/source checks, seal, submission and data acceptance; the package owner performs no platform/GPU operations.
- CPU/package acceptance: false reaches all four native node plans and argv unchanged; original true-toggle reproduces the table-related timeout control while false preserves export and succeeds; export-too-slow still fails. Keep original profile03 failed artifacts. Require dual review and actual staged-entry/image controls before launch. No new flag, timeout scaling, profiler rewrite, numeric mapping, capture reduction or admission relaxation is allocated.
- Runtime acceptance remains all432 client requests successful, four actual hosts/32 H800 identities,>=3 useful kernel-bearing steps per role/lane, published formal metadata/no unexpected losses, raw DP pairing, stream correlation and post-stop evidence. The optional `profiler_out_*.txt` files are intentionally absent; complete raw traces and metadata remain required. Export is still synchronous: if it exceeds the native budget, preserve evidence and stop after this attempt rather than retry automatically. Captured per-operator ownership and serving/operator-cost admission remain independent and unverified.

### D-V4-PROFILE03-METADATA-CENSUS-FIX — offline artifact classification

- Root directly confirmed64 `torch_traces/` entries are32 `.pt.trace.json.gz` plus32 `profiler_out_*.txt`, not64 raw traces. The metadata adapter classified by directory only. Correct only this artifact census in a new pinned copy (<=4gross code lines), retain original script/output bytes in scratch, rerun the full190-file/32-rank metadata audit and distinguish32 raw traces/32 optional tables. Other output fields must remain equal apart from explicit corrected census/provenance; stale/formal/DP-count and FAILED limits stay unchanged. Update only existing `failure/{metadata_audit.json,metadata_audit.md,metadata_command.md}` (<=20gross maintained Markdown) with original/corrected hashes and exact commands. No runtime/GPU/native/package or parent-document edits.

### D-V4-GROUNDTRUTH-FIXTURE — independent CPU regression restoration

- [Decision — 2026-10-24] Under D-V4-QUEUE-PARALLEL-WORK and D-V4-BOUNDED-DIAGNOSTIC-AUTHORITY, restore the known pre-existing positive-fixture failure recorded in `profile01/dp_sequence/baseline_diagnosis.*` and `profile01/integration_cpu/`. Baseline44f2ddb9 and candidate178c1123 produced identical32counter/hook dictionaries and32missing-dp_utils setup errors; integrated1437ace6 retained465PASS/1unchanged-baselineFAIL/0SKIP. Preserve all original failures.
- One owner may edit only `tests/unit/test_native_groundtruth_check.py`,<=60gross changed lines, plus its own records under `profile01/groundtruth_fixture/`. Reuse the actual frozen R3 successor's source-backed V2 classifier/DP-dispatch stand-ins and an explicit CPU group. Do not add an empty dp_utils module, remove observed hooks, relax assertions, alter a frozen checker, edit runtime/native/C26/core files, or modify any sealed package. If the complete restoration needs more scope, return the inspected reason rather than compressing or changing behavior.
- Acceptance: exercise the actual32-rank positive path and retained negative/oracle controls, run the complete fixture-enabled groundtruth-check suite with0failures/0errors/0skips, verify immutable checker/runtime/native/package hashes and bind the existing77-case unchanged-domain matrix. Root independently reruns affected checks and performs phase-level review before integration. Tests must use the owner-verified asyncopd Python3.12 environment and the actual existing fixture paths; no new dependency installation or fabricated reference inputs.
- Source/test budget is separate from raw receipts/diffs and maintained task Markdown<=200lines. Owner commits verified source and records in its local worktree and stops. No GPU/platform/auth/Docker/network/control operations or further delegation; root keeps profile04's existing launcher alive and owns its actual acceptance.

### D-V4-PROFILE04-STAGE-DOCS — submission milestone synchronization

- One records owner may update only parent `task_2026-10-15_glm53_dsv4_vllm028_h800/{progress.md,summary.md,plan.md,tickets/05-dsv4-flash-family.md,tickets/09-predictor-bundle.md,v4_prediction_next_20261022.md}` and this task's `summary.md`,<=150gross maintained lines. Read the frozen profile04 root records atf4f4ab79; report one submitted jobexp-1024-172225-512616 and the dated Starting/four-unassigned snapshot, not runtime success. Preserve original profile03 FAILED evidence, limited callback/isolated acceptances, native06R4, full05/09/19OPEN, GLM pause and CPU-overhead-last order.
- Check new links and immutable evidence bindings; do not edit primary requirements/plan/progress or any JSON/source/package/run output. The root-owned primary progress already records the current submission. No Git staging/commit, platform/network/credential operations or subagents. Return changed paths, gross count, verification and limitations; root owns acceptance and commit.

### D-V4-NATIVE-CAPTURE-SCOPE — read-only ownership prerequisite

- Continue the already planned native ownership assessment in parallel: inspect whether existing Python per-operator/layer scopes are reachable during native FULL and breakable/PIECEWISE graph capture after compilation, and what the existing capture/replay hooks actually cover. One read-only source owner may write only `profile01/native_capture_scope.md`,<=110maintained lines, with local-source hashes/line references and explicit source-only limits. No source/test/package/checker edit, CUDA callback subscription, probe implementation, platform/Docker/GPU/network or Git operation. Preserve sealed profile04. Distinguish direct scope witnesses from creation order/labels; mark runtime execution and any non-image-bound Torch claims UNKNOWN. Return the smallest evidence-backed next proof, not an implementation allocation or automatic run.
