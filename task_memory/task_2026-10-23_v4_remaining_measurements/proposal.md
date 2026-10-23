# V4 remaining measurements: grounded minimal proposal

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Completed the single-window full-native resource proposal after G2 source 05952396 acceptance; GPU approval remains pending. |
| 2026-10-23 | YC approved this run and standing task-required H800 RJob authority; the first run retains this concrete resource/window plan. |
| 2026-10-23 | Selected fresh profile02 after the unsubmitted profile01 entry-import failure; resources, workload and collection window are unchanged. |
| 2026-10-23 | Checkpoint correction: CUDA_EVENT component sums now exclude the outer scope (audit receipt); J3 premise rewritten for active breakable capture in native V4. |
| 2026-10-23 | YC approved the recommended KERNEL_ONLY CPU implementation/design route; requirements.md now owns the decision. GPU figures below remain historical preliminary estimates. |
| 2026-10-23 | Created. Read-only design legwork for a user GPU/modelling decision: existing producer APIs, exact capture limits, two options for the serving attention total, minimum jobs and per-job closure. No source, test, config, GPU, platform or cloud change. |
| 2026-10-23 | Root-acceptance correction, five items: replay measurement is a current-recorder limit, not an impossibility; kernel-to-scope disjointness is not completeness or temporal non-overlap, so additivity needs coverage plus stream-dependency evidence; J4/J5 do not establish per-role TP8/DP2/EP16 (16 GPUs per role, 32 for P0); J3 feasibility unproven from the DSA/V1 adapter and the target window differs from the existing witness; resource figures relabelled preliminary partial-campaign estimates, not an executable campaign or approved budget. Bundle restated as V4-private approved-shaped. |

Task ID: `W128-V4-REMAINING-MEASUREMENT-PROPOSAL`. Branch `feat/glm53-dsv4-vllm028`, HEAD `e89b45d`.
Historical proposal source: e89b45d. Current native06 is ACCEPTED_NATIVE_REFERENCE_R4,
archived at972f0cb1 after frozen8/8PASS and208tests. Nothing here re-opens R4 or the stopped
CUPTI investigation (3/3 attempts spent).

## Main proposal

**Option A is approved subject to coverage and GPU-dependency proof.** Current
`requirements.md` records YC's “批准，继续” under D-V4-KERNEL-ONLY-GPU-COST. Parent D-Q10
already authorizes private measured-input wiring. This selects per-operator kernel costs,
not a direct sum of scope medians or a CUDA_EVENT residual assigned to CPU.

**What the accepted data supports — and what it does not.** CUDA_EVENT component medians summed
excluding the outer `dsv4_attention_layer` scope, versus the composite median (ms): csa layers2/4
3.887168/3.327680, 3.874048/3.312032; hca layers3/5 1.635584/1.324704, 1.620832/1.310880;
uncompressed layers0/1 1.484800/1.524128, 1.014272/1.034080 (`audit/test_report_2026-10-23_kernel_coverage.md:126-130`,
`audit/kernel_coverage_receipt_r2.json`). The earlier 7.2148/2.9603/3.0089 figures double-counted the
outer scope and are historical. KERNEL_ONLY sums: 3.3795, 3.3402, 1.3159, 1.3209, 0.9699, 0.9786.
The audit also finds owned-kernel sum above interval union for compressed layers, so scopes overlap in time.
`dsa_kernel_trace.py:63-131` assigns each correlated kernel to exactly one host-launch annotation, so
no kernel is counted twice. That is double-count freedom only: uncorrelated kernels are silently
dropped, so it proves no completeness, and two scopes can overlap in wall-clock time across streams
while each still owns distinct kernels. The ~2% csa/hca agreement is suggestive, not sufficient; an
additive wall-clock phase vector additionally requires (a) source/trace coverage that every attention
kernel in the window is attributed to a scope and (b) an explicit GPU stream-dependency/overlap
account. The uncompressed 0.9699/1.5241 difference stays unexplained; it is a
wall-clock-versus-occupancy attribution gap, not a CPU cost, and must not be closed by scaling.

**Option A** is that vector, fed through the existing V4-local phase table (`deepseek_v4.py:20-63`),
subject to (a)/(b). **Option B** is the composite `dsv4_attention_layer` elapsed with aux overlap
declared unmodelled: obtainable from replay, but discards all 29 named operators and gives no
per-operator attribution.

**Two distinct consumables, not one.** The current recorder supports CUDA_EVENT during replay:
`dsa_full_dummy.py:149` asserts replay executed no Python scope, and `:155-164` emits per
(layer, scope) CUDA_EVENT times from capture-time external events. It does **not** support per-scope
KERNEL_ONLY there: kernel-only attribution depends on host-launch annotations absent from a single
graph launch. That is a current-mechanism limitation, not proof of impossibility. So (i) replay
CUDA_EVENT per scope and (ii) eager KERNEL_ONLY reference are separate inputs, and graph/phase
equivalence compares (i) against (ii).

**Campaign scope and what each job does not prove.** Preliminary partial estimates for feasible
pieces; not an executable complete campaign, not an approved budget.

| Job | Covers | Prerequisite | Preliminary ceiling |
| --- | --- | --- | --- |
| J1 | tail operators `dsv4_embedding`, `dsv4_mhc_*`, `dsv4_final_norm`, `dsv4_lm_head` | `patch.object` hooks on free functions (`model.py:1141,1176,1189,1203`); `_wrap` cannot reach them | 1 GPU, <=1 GPUh |
| J2 | decode per-scope at cached 6143..7166, width 1 and 128 | V4-private case builder: `dsa_plan.py:625-660` fixes `batch_size=1`, `max_num_seqs=1`, and the engine is TP1 (`dsa_vllm_runner.py:614`) | 1 GPU, <=2 GPUh |
| J3 | decode replay per-scope timing versus eager reference | Native V2 profiling extension (plan step H); breakable capture is active, so eager attention segments re-run Python at replay; capture-time event survival is a runtime gate (below) | 1 GPU, <=1 GPUh, conditional |
| J4 | routed compute at EP16 | `frontier.profiling.moe.main --expert_parallel_sizes 16 --load_distributions balanced`; its `--num_tensor_parallel_workers` is the MoE GEMM shard (`moe_wrapper.py:204`), not serving attention TP | 16 GPUs, <=12 GPUh |
| J5 | per-role TP8/DP2/EP16 | 16 GPUs per role, 32 for P0; use the existing full native worker/extension path. A new shared TP parameter is not a prerequisite. Scoped V2 profiling/capture/runtime proof is still needed | unestimated |

J4 alone does not prove TP8/DP2/EP16, and a TP8-only 8-GPU run cannot prove DP2 or EP16, so neither
closes P0 topology. P0 requires per-role 16 GPUs.

**J3 feasibility is not proven.** The DSA statement "replay executes no Python scope"
(`dsa_full_dummy.py:149`) belongs to the non-breakable whole-model `CUDAGraphWrapper` path. It does
not hold for native V4: all four native06 `server.log` line16 record `Auto-enabling
VLLM_USE_BREAKABLE_CUDAGRAPH=1` (`vllm/config/vllm.py:1313-1336`), `@eager_break_during_capture`
(`attention.py:405`) makes the attention core an eager segment, and
`BreakableCUDAGraphCapture.replay()` re-executes those callables (`breakable_cudagraph.py:195-217`).
Python annotations therefore run during decode replay for eager segments; captured-segment kernels
still have no Python scope (`native_profile_seam.md` §4–§5). Whether capture-time event pairs survive
torch.compile and breakable capture unchanged stays a runtime gate (§8). Native06 decode used FULL and
PIECEWISE descriptors 1/2/4/8/16 with 1..11 scheduled requests per DP lane (all 32 rank files,
`native_profile_seam.md` §3); per-step `has_prefill` classification is required and shape-key reuse
does not qualify context6143..7166 timing.

**Exact refusals retained.** Absent rows, absent layout contract or absent provenance still raise
`DeepSeekV4CoverageError`; nothing here relaxes a guard.

## Evidence table

| Fact | Source |
| --- | --- |
| Q10 binds shared interfaces; measured consumption approved; bundle is V4-private approved-shaped | `requirements.md:999`, `:1077-1085`; `dsa_full_dummy_input.py:17,29-32,50-54` |
| KERNEL_ONLY is the uninstrumented reference; CUDA_EVENT outer wraps inner scopes | `test_report_2026-10-21_reference_event_audit.md:29-35` |
| DSA non-breakable replay runs no Python scope; that path emits CUDA_EVENT records. Native V4 uses breakable capture whose eager segments re-run at replay | `dsa_full_dummy.py:149`, `:155-164`; `native_profile_seam.md` §4 |
| Kernel→scope assignment is single-launch, single-scope; no completeness claim | `dsa_kernel_trace.py:63-131` |
| Existing FULL path is DSA/V1-only with uniform query-one witness | `dsa_full_dummy.py:90-99`, `:20-21` |
| Producer engine is TP1/DP1; identity asserts `engine_tensor_parallel: 1` | `dsa_vllm_runner.py:614`, `dsv4_attention_main.py:157` |
| Producer engine forces dense MLP, so MoE cost cannot come from it | `dsa_plan.py:321-325` |
| MoE profiler TP parameter is a GEMM shard, not serving attention TP | `moe_wrapper.py:204` |
| P0 per-role topology TP8/DP2/EP16 = 16 GPUs per role | `requirements.md` D-CALIBRATION-PRIORITIES |

## Approved CPU work and later run decision

Option A CPU implementation/measurement design is approved. Input wiring uses the private
sidecar pattern; full-layout instrumentation can reuse native_pd_worker/runtime extensions.
`native_profile_seam.md` records the source route and its unresolved runtime proofs. Root's
actual native06 checks of all 32 rank files find prefill NONE (one request/6144 tokens) and decode
FULL plus PIECEWISE at descriptors 1/2/4/8/16 (1..11 scheduled requests per lane). A128-request captured descriptor is capacity,
not an observed serving batch; shape-key reuse is not context/phase timing qualification.

Every new run needs a concrete validated implementation and run plan. Task-required H800
resources now use D-V4-H800-RJOB-AUTHORITY; the preliminary table remains a historical estimate.

## Unsupported facts and blockers

- Completeness of kernel→scope attribution and stream-overlap accounting are both unproven.
- V4 FULL replay per-scope timing at real cached context is unproven; J3 may fail on witness design.
- EP16 all-to-all/dispatch costs are collective measurements, outside J1-J5.
- The uncompressed 0.9699/1.5241 gap is unexplained; no scaling is proposed.
- No CPU-overhead measurement is proposed; it stays last.
- The GPUh figures are planning ceilings for feasible pieces only, derived from native06's observed
  ~12 GPUh as an order-of-magnitude reference. They are not a measured cost or an approved budget.

## G2 single-window collection — approved

The next diagnostic collection uses the existing full native P/D path. G2 source `05952396` is CPU-accepted after both independent re-reviews and final verification (`driver/primary_acceptance.json`). The earlier J1–J5 table remains a historical list of coverage needs, not an executable campaign or approval.

| Item | Proposed value and evidence |
| --- | --- |
| Resource ceiling | One job with four nodes, each with 8 H800 GPUs, 64 CPUs and 512 GiB RAM; 32 GPUs total, `codesign`, non-preemptible. Payload limit 3600 s, startup limit 1800 s, replay limit 1800 s; at most 32 GPUh. Native06 used 12.002222 GPUh; profiling duration is unmeasured. |
| Topology | P16+D16; each role TP8/DP2/EP16/PP1, one local DP lane per node. Preserve native V2,MXFP4/MARLIN,balanced_topk and native06's graph configuration. |
| Native/image | Source2cf0a6915ce544dc493a0990f2ea38d81601128a; image`hub.i.basemind.com/vllm-openai/vllm-openai@sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635`. Re-verify source and binary pins in the new package before launch. |
| Frozen workload | Existing canonical6144/1024,128formal requests plus16warmups per round,three rounds,QPS1. No request/arrival changes. This instrumented run cannot replace native06's clean reference. |
| Input provenance | `driver/proposal_inputs.json` rehashes the five workload files against `FINAL_FREEZE_R4.json`; trace CSV SHA-256 is `2d557d7d923b71fbf62bc7e552d1fb547a8ec88c1e54660c0c76cc2de3c49ec2`. The three native06 configuration files are reference inputs only; the new package must explicitly generate its profile configuration and source/output bindings. |
| Engine limits | Preserve max_model_len7168,max_num_batched_tokens8192,max_num_seqs128. These are configured capacities, not runtime batches. |
| Window | Both role-head APIs bracket first-formal replay and its existing drain. Native `max_iterations=64`, zero delay/warmup/wait and `ignore_frontend=true`. One native torch/CUPTI collection window in this run; any necessary subsequent H800 run gets a separate concrete plan under the standing authority. |
| Trace options | Torch CPU+CUDA activities from native Worker.profile; gzip enabled, stack/shapes/FLOPs/memory profiling disabled. Raw traces remain necessary for kernel correlation and coverage checks. |
| Collector limits | Per rank: at most 250000 retained rows and 268435456 serialized row bytes, separately for startup and the selected window. Retain the startup archive, window snapshot and final inventory. These are capacity limits, not guaranteed coverage; drops and archive failures remain explicit. |
| Storage | Planned new tag `dsv4-h800-p0-profile-20261023-02`, under `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18/dsv4_h800_p0_profile_20261023`. The unsubmitted profile01 package is retained after its entry-import failure. Each node writes traces under its role/DP subdirectory. Verify the new destination is unused; mount the current local workspace and personal cloud volume through StepMind Python. |

The 64-step choice is a finite initial observation window, not proof of three aligned useful batches. Native iteration limits also count zero-token lane steps; retain those records and validate coverage using the matched distributed execution. Native06's actual shapes were 6144/NONE for prefill and 1/2/4/8/16 under FULL/PIECEWISE for decode, with at most 11 scheduled requests/tokens per lane (`driver/native06_window_basis.json`). Report the tuples actually captured. This window does not cover the complete decode context 6143..7166 range.

### Artifact-volume accounting and remaining uncertainty

At the proposed row-byte limit, one32-rank snapshot has at most8GiB of serialized row payload. Startup archive, stopped-window snapshot and final snapshot could together retain at most24GiB of row payload. This excludes metadata, Python object overhead, formatting and torch trace files; it is not a host-memory or total-storage bound. Gzip trace size remains unmeasured for full-model V4 and must be reported separately rather than inferred from the isolated six-layer trace. No trace-size validator may interrupt healthy native work.

### Runtime acceptance and source prerequisites

1. CPU source gate complete at `05952396`: both reviews, the cold-start correction, 433/750-line accounting, separate final suites and explicit matrix binding passed. Pin this source and the frozen workload hashes in a fresh package.
2. Verify the new staged extension, real native hook resolution, per-node trace paths and all image/source/binary pins. Resource authority is D-V4-H800-RJOB-AUTHORITY; package verification does not itself qualify GPU evidence.
3. At runtime retain all32worker receipts, actual model/layout/routing/source identity, local scheduler requests/tokens, DP-synchronized descriptor/mode, first-formal IDs and actual start/stop/drain outcomes. At least three distinct aligned useful logical batches are required for each claimed phase; absent or dropped tuples remain unverified.
4. For a usable KERNEL_ONLY slice, prove kernel-to-launch/owner completeness, event handles/generations and stream mapping, and preserve actual PDL/GDC evidence. Missing captured-segment ownership or stream dependencies prevents additive totals; no scaling or CPU-gap attribution fills the difference.
5. Archive raw traces, inventories, client rows, terminal receipts and every observation failure. Auxiliary profiling/validation errors stay soft; healthy workload completion continues, affected evidence remains UNVERIFIED. Any later collection records its need and concrete resources; task-required H800 runs use the standing authority.

Current status: **SEALED_READY_DIAGNOSTIC**, decision D-V4-H800-RJOB-AUTHORITY. The original profile01 entry failure is preserved. Fresh profile02 passed both repair reviews and root's final actual-entry/image/source checks after merge `cc0227e9`; its 45 inputs are sealed by manifest `51e39a42ec863e496f20be8a165230e0dee17c34906ad6bd4fade14bf898ee91`. Fresh quota/destination checks are recorded in `profile01/test_report_2026-10-23_root.md`; root submission is next. Runtime and evidence qualification remain open.
