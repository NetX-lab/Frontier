# V4 remaining measurements: grounded minimal proposal

## Modification History

| Date | Change |
| --- | --- |
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
PIECEWISE with at most11 scheduled requests on the16-token descriptor; per-step `has_prefill`
classification is required and shape-key reuse does not qualify context6143..7166 timing.

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
actual native06 rank0 checks find prefillNONE (one request/6144tokens) and decodeFULL plus
PIECEWISE (maximum11scheduled requests/tokens). A128-request captured descriptor is capacity,
not an observed serving batch; shape-key reuse is not context/phase timing qualification.

New GPU resources will be requested only with a concrete validated implementation and run
plan. No shared-field decision or new job budget is implied by the preliminary table.

## Unsupported facts and blockers

- Completeness of kernel→scope attribution and stream-overlap accounting are both unproven.
- V4 FULL replay per-scope timing at real cached context is unproven; J3 may fail on witness design.
- EP16 all-to-all/dispatch costs are collective measurements, outside J1-J5.
- The uncompressed 0.9699/1.5241 gap is unexplained; no scaling is proposed.
- No CPU-overhead measurement is proposed; it stays last.
- The GPUh figures are planning ceilings for feasible pieces only, derived from native06's observed
  ~12 GPUh as an order-of-magnitude reference. They are not a measured cost or an approved budget.
