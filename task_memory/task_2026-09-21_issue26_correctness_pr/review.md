# Issue 26 Correctness PR — Review Records

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-21 | Created with the pinned source snapshot. |
| 2026-09-21 | Step 1 complete: candidate and vLLM audits landed, dispositions recorded, two decision checkpoints raised. |
| 2026-09-21 | Corrected the W3 and W4 rows: the step-id namespace is not partitioned by sync kind, only the open-step binding table is. Verified against `forward_sync_state.py` at `c18eb2c`. |
| 2026-09-22 | W3 delivered and measured; R35-02 closed; the unreleased multi-lane monolithic MoE shape recorded as an open item. |
| 2026-09-22 | Recorded the maintainer's PR #34 / PR #35 review: D1 and D2 resolved, ten review comments dispositioned, each verified against source. |
| 2026-09-22 | Self-review of that record: corrected the lockstep mechanism and counter semantics under D1, the line references under R34-01, the R34-03 remedy (the baseline label is itself an assembled partial run), and added the omissions listed under "Found on re-review". |
| 2026-09-22 | W5 closed without source changes. Corrected the Mechanism A premise: the routing distribution has no per-role override on main, so both W5 mechanisms are unreachable from any released configuration. The user chose to keep the single global field; the drafted implementation was reverted and archived as a patch. |

## Pinned source snapshot

| Repository / reference | Revision | Role |
| --- | --- | --- |
| `NetX-lab/Frontier` `main` | `1f694f7c549aa3aeeb7c5bbae04e119c09167a77` | Integration baseline (verified 2026-09-21 after `git fetch`). |
| `NetX-lab/Frontier` `bug/ttft-check` | `a7b3320fe9b8b083ee86b91dae3d6838f4443d91` | Candidate. Final commit touches only `task_memory/`. |
| Candidate parent before evidence import | `b7f8d055461d9208b8ae57eceeb6c0246cbc8d3c` | Source comparison point (verified equal source tree to `a7b3320`). |
| Merge base | `d71ad80b0800880808a0857fd30477e6d96592c6` | Verified with `git merge-base`. |
| `fwyc0573/vLLM-BS` | `ea95f571e20937c7c908c6d59ddd1cd6bf9268f1` | vLLM reference, `.real-engine/vLLM-BS`. |
| Upstream `vllm-project/vllm` `v0.10.2` | `01efc7ef781391e744ed08c3292817a773d654e6` | Resolved by fetching the tag into the reference checkout. |
| `fwyc0573/frontier-htsim` main gitlink | `b8518afcc310f0fe0e3ce52ba6b4f0bf57a3be04` | Current optional backend. |
| Candidate gitlink | `e564935d3874d8c71b52a554ab7c9a72e5e19f68` | Not reachable on the configured remote (HTTP 422, 2026-09-21). |

### vLLM reference relationship to v0.10.2

The fork's HEAD is a direct descendant of upstream `v0.10.2`: the merge base equals the tag commit, and the fork is 40 commits ahead, 0 behind. It pulled nothing from upstream after branching.

Decisive for this PR: `vllm/v1/engine/core_client.py`, `vllm/v1/engine/coordinator.py` and `vllm/forward_context.py` are **byte-identical to v0.10.2**, so the DP placement and count-publication behavior read for work package W4 is upstream behavior, not fork behavior. `vllm/v1/core/sched/scheduler.py` and `vllm/v1/worker/gpu_model_runner.py` do differ; the differences are characterized per file in `reference_vllm_0_10_2.md`.

## Supporting audit reports

| File | Contents |
| --- | --- |
| `audit_scheduler.md` | Three-way source audit of the candidate's scheduler changes (W2, W3, W4). |
| `audit_predictor_profiling.md` | Three-way source audit of the candidate's predictor and profiling changes (W5, W6). |
| `reference_vllm_0_10_2.md` | vLLM reference-behavior tables with pinned file and line citations. |

Claims in those reports that this PR depends on were re-verified directly against the source before being recorded here; the five spot checks are listed in `validation.md`.

## Work package dispositions

### W2 — Round-robin DP placement rotation

| Item | Finding |
| --- | --- |
| Defect on main | Present. `frontier/scheduler/cluster_scheduler/round_robin_cluster_scheduler.py:385-386` assigns the DP lane with `dp_id = local_idx % self._replica_dp_size`, where `local_idx` enumerates only the requests handled in the current call. The persistent `_request_counter` is used for the replica index (`:372`, `:376`) but not for the lane. |
| Main already has the correct formula | Yes, for one role only. `_schedule_decode_lane_round_robin` at `:438-439` computes `replica_idx = (counter + idx) % n` and `dp_id = (counter + idx) // n % dp_size` from the same persistent counter. `schedule()` routes that method only for `ClusterType.DECODE` (`:75-76`); `_schedule_batch_mode` is the `else` branch (`:85-86`) and serves MONOLITHIC and PREFILL. |
| Disposition | **ADAPT.** The fix is an internal-consistency repair using main's own formula, not a new policy. |
| Caveat to resolve during implementation | `_schedule_batch_mode` returns results grouped per replica; the decode variant returns arrival order. The two cannot simply be merged until the consumers of that ordering are checked. Recorded as an open item. |
| Planned test | Extend `tests/unit/test_cluster_scheduler_dp_lanes.py`: same ordered request stream split across different call boundaries must give the same request-to-lane assignment; cover DP1, DP>2, multiple replicas, non-contiguous replica ids, an empty call, and continuation after it. |

### W3 — Shared monolithic forward completion

| Item | Finding |
| --- | --- |
| Defect on main | Present at three layers. `frontier/events/replica_stage_schedule_event.py:157-184` selects the prefill or decode sync path from the lane's own `num_prefill_tokens`; `frontier/scheduler/utils/sync_state.py:29-34` allocates two independent waiting rooms for MONOLITHIC; `frontier/scheduler/utils/forward_sync_state.py:38` partitions the open-step binding table `_open_steps_by_kind` into `"prefill"` and `"decode"`. A mixed-phase forward puts one required lane in each room, neither peer is idle, and both rooms stall. **Correction to an earlier version of this row:** the step-id *namespace* is not partitioned. `_next_step_id_by_replica` is keyed by replica alone (`:42`, `:133`, `:139`, `:163`), so allocation is already Replica-scoped and monotonic across both kinds. What is partitioned is the binding table and the waiting room, which narrows the change W3 has to make. |
| Clean ports | The `"forward"` sync kind, and the cross-lane duplicate-request guard in `frontier/scheduler/utils/ep_wave_inputs.py`. |
| Blocked hunk | The candidate's decode final-metrics hunk calls `scheduler._create_corrected_execution_time_for_metrics(...)`, which **main deleted**; `_create_prefill_corrected_execution_time_for_metrics` also changed signature and now lives at `base_cluster_scheduler.py:1138`. Porting verbatim raises `AttributeError`. Main additionally added `metrics_store` and `ep_wave_reporting_enabled` plumbing to `ep_wave_schedule.py` and `prefill_collective.py` that the candidate lacks. |
| Disposition | **ADAPT** for the lifecycle change as one coherent unit; **BLOCKED** for the metrics hunk until it is rewritten against main's current execution-time ownership. |
| Planned test | The behavior matrix in `plan.md` §9, including at least one test that drives the real event loop with overlapping prefill and decode and injects deterministic times only at the predictor boundary. |
| Delivered | `65ed8a7`. Lifecycle adapted as one unit; the metrics hunk (I8) stayed **BLOCKED** and is now recorded as a deliberate exclusion in `design.md` rather than an open item. 23 behavior tests plus the real-runtime fixture; four deliberate-defect controls each fail for their own reason. Fidelity matrix 71 of 71 identical, matching the expectation recorded before the run. See `test_report_2026-09-22_w3_shared_monolithic_forward.md`. |

### W4 — Opt-in vLLM-style DP placement

| Item | Finding |
| --- | --- |
| New public surface in the candidate | `ClusterSchedulerType.VLLM_LOAD_BALANCING`, the CLI token `--cluster_scheduler_config_type vllm_load_balancing` (no other new flag), `VllmLoadBalancingClusterSchedulerConfig`, a new `frontier/config/cluster_scheduler_config.py`, a registry entry, `RequestLoad`, `VllmDPLoadBalancer`, `BaseClusterScheduler.schedule_at` / `on_replica_batch_end`, and `BaseReplicaScheduler.get_request_load`. |
| Reference semantics confirmed | Engine selection scores `waiting * 4 + running` in the **frontend**, `vllm/v1/engine/core_client.py:1146`. The coordinator transports unweighted `[waiting, running]` pairs. Timing constants confirmed: `min_stats_update_interval_ms = 100` (`coordinator.py:116`, a floor rather than a period), the bare literals `5000` (`coordinator.py:198`) and `50` (`coordinator.py:202`). The specification's grouping of the weight 4 with the coordinator constants is corrected here: it belongs to the frontend. |
| Request populations confirmed | Admitted-but-not-yet-scheduled requests count as **running** (`scheduler.py:812-813`, appended inside `schedule()` before the model runs); preemption moves running to waiting within the same step (`:507`, `:537-538`); requests finishing this step are in neither (`:1381-1385`, before `make_stats` at `:1413`). |
| Report suppression confirmed | Two independent stats producers exist; only the DP one feeds the coordinator, and it is suppressed whenever the count pair is unchanged (`core.py:1081`). That is why the 5 s heartbeat exists and why the step counter is a sparse tag rather than a dense counter. |
| Step-identity problem | The candidate uses `ForwardSyncState.get_step_id(batch)` as the report-order key (`vllm_load_balancing_cluster_scheduler.py:52-56`, latched at `vllm_dp_load_balancer.py:64-67` against a single scalar shared across engines). On main that identity is **invalid for dense models** (`base_replica_scheduler.py:459-466` sets `_forward_cohort_id` from a per-DP-lane creation counter that dense never promotes to a Replica-scoped counter, and the policy's guard does not require MoE), **invalid for MoE with DP>1 until W3 lands**, though for a narrower reason than first recorded: the ids come from one Replica-scoped counter, but the two lanes bind into separate per-kind open-step tables, so a mixed-phase forward never reaches a single shared step to key a report on, and valid for DP1. Idle participants are never reported. The id also advances per layer (`sync_entry.py:64`, `forward_sync_state.py:133-139`), so it is monotonic but is not a vLLM step counter. |
| Disposition | **BLOCKED on W3**, then ADAPT. This makes the execution order W2 (independent) then W3 then W4. |
| Decision raised | See D1 below. |
| Delivered | `10dd474`. The balancer models the frontend's selection and the coordinator's publication schedule separately, each constant cited against vLLM v0.10.2. The step-identity problem is resolved by measurement rather than by assumption: MoE keys are monotonic per Replica after W3, dense keys interleave under staggered arrivals, so the constructor rejects dense above one lane. 61 unit tests, 3 real-runtime cases including a placement that diverges from round-robin, five deliberate-defect controls each failing for its own reason, and a 71-of-71 identical fidelity matrix. See `test_report_2026-09-22_w4_vllm_dp_placement.md`. |

### W5 — Routing load distribution versus routing implementation identity

| Item | Finding |
| --- | --- |
| Defect on main | **Corrected 2026-09-22: present as latent code, not reachable.** Both mechanisms below need two clusters that resolve different routing paths in one run. On main and on this branch, `moe_routing_distribution_type` is declared once, on `ReplicaConfig`; `ClusterConfig` declares no `{prefill,decode,decode_ffn}_replica_config_moe_routing_distribution_type` field, so `get_field_value("moe_routing_distribution_type")` (`cluster_role_config.py:58-63`, `:73-75`, `:87-89`) always takes its `getattr(..., None)` miss and returns the global value, and `python -m frontier.main --help` generates exactly one flag, `--replica_config_moe_routing_distribution_type`. Every cluster in a run therefore resolves the same path and neither mechanism can fire. The original row, kept below for the record, took the `get_field_value` lookup as evidence of a declared override. |
| Mechanism A (as originally recorded; premise corrected above) | Routing distribution is settable per role (`config.py:4242-4247`, `:4257-4273`, `:4302`), so a PDD run can use `PREFILL=balanced` and `DECODE=random`, which resolve to different runtime paths (`moe_routing_runtime.py:29-32`). `trained_model_signatures` is one set shared across clusters (`shared_prediction_model_manager.py:702`, `:707`, `:778`) and `ffn_signature` (`:1377-1382`) carries **no routing term** (re-verified). The second cluster therefore returns early at `:1384-1386`, before `_validate_moe_dataset_contract` at `:1433-1439`, so there is no fail-fast, and it predicts `moe_gating_routing_topk` using the other cluster's estimator through `_models_view_for_family` (`:4358-4388`), which matches on layer identity alone. |
| Mechanism B | When both clusters do train, `_store_model_precision` keys on `(model_name, identity)` at `:4238` with a layer-shape-derived identity, so the second model overwrites the first. |
| Where identity does reach today | Only dataset row selection (`:1294-1326`, `:1235-1265`) and the per-call `moe_df_cache` key (`:1486-1505`), which is scoped to a single cluster call and therefore never prevents the collision. |
| Correction to the specification's premise | The **persistent disk cache is not the hole**. `_get_model_hash` (`:4008-4056`) hashes `df.to_json()` at `:4033`, and `_load_moe_df` filters rows without dropping columns (`:3628`, `:3712-3720`), so cached artifacts are separated incidentally. |
| Disposition | **PORT** the resolver override, the config field and its copy, the predictor helper, dataset admission, the training signature term, and the per-model training identity. **ADAPT** the registry key widening: the candidate's helper rewrite enumerates only `eager` and `kernel_only` and would break main's third `device_event` measurement family (`:4160-4162`, `:544-551`). |
| Two implementation caveats | The candidate appends `_routing{path}` to the whole `ffn_signature`, which separates every MoE FFN model rather than only the routing-topk model. That is correctness-safe but over-broad and should be narrowed. Main's standalone trainer already calls this concept `routing_runtime_path` (`frontier/training/cli.py:153`), so the candidate's `moe_gating_routing_runtime_path` would be a third spelling; one name must be chosen before any public flag exists. |
| What the override would have decided | One thing only: which `moe.csv` rows train `moe_gating_routing_topk`, i.e. whether that operator's predicted cost is the fused-topk kernel's or the profiler's uniform round-robin path's (`frontier/profiling/moe/moe_impl.py:75-103`, `:195-225`). It never touched expert load, grouped GEMM, EP synchronization or scheduling. Measured on the three checked-in `moe.csv` datasets that carry both routing paths (`data/profiling/compute/a800/qwen3-a3b-30b-moe`, `h800/Phi-tiny-MoE-instruct`, `h800/step-moe-noquant-small`): matched-feature `moe_gating_routing_topk` medians are 0.073 / 0.189 ms, 0.050 / 0.082 ms and 0.041 / 0.072 ms (standard / uniform), and the difference is 3.1% (median; 24% at the smallest token counts), 7.3-7.8% and 4.2-5.4% of the summed per-layer operator medians. That is the cost of choosing the wrong path, and it is why the existing distribution-to-path mapping stays. It is not a fidelity change attributable to the override: with the field unset, every configuration resolved exactly as before. |
| Final disposition (2026-09-22) | **NOT PORTED.** The user decided to keep the current contract -- every cluster reads the one global `moe_routing_distribution_type` and resolves one routing path per run -- rather than add the override that would make the collision reachable and then the registry axis that would defend against it. The drafted implementation (global field, three per-role overrides, four CLI flags, single-owner resolver, routing-aware training signature and family gate, a `(model_name, identity, routing_runtime_path)` registry key with legacy-conflict rejection; 745 patch lines, never committed) is archived at `w5_reverted_moe_routing_runtime_path.patch` for reference. Nothing in `frontier/` changed for W5. The retained candidate tests `test_moe_routing_runtime.py` / `test_moe_routing_runtime_model_sharing.py` listed under "Retained from the candidate's tests" are therefore not adopted either. |

### W6 — Legacy fused-MoE profiling arithmetic

| Item | Finding |
| --- | --- |
| What main's legacy path omits | Exactly two steps. `_run_fused_moe_iteration` (`frontier/profiling/moe/moe_vllm_kernel.py:365-431`) takes a bare first-half slice at `:404-405`, discarding the up-projection half instead of applying gated SiLU, and performs no local top-k reduction (`moe_sum` appears nowhere in the file). FP8 activation quantization (`:407-413`) and routing weights on the second GEMM (`:423`) are already correct; only the operand is wrong. |
| No conflict with main-only work | The functional and MXFP4 branch returns at `:869`, before the legacy allocations at `:904`. |
| Required adaptations | Port hunks, not the file: a whole-file take would revert 419 lines of main-only work (the functional `fused_experts` adapter, MXFP4/AITER, `device_event` timing, profile-method platform validation). Move `from vllm import _custom_ops as ops` out of the top-level `try` so it cannot perturb main's two-branch API detection (`:116-140`). Reuse main's existing `SiluAndMul` wrapper (`frontier/profiling/common/layers/activation.py:8-34`) rather than the raw op. |
| Measured scope today | The whole `_step` body, timed either with per-iteration CUDA events and a synchronize (`:603-626`) or through `record_function("vidur_moe_grouped_gemm")` (`:629-653`). All buffers are allocated outside the timed region. |
| Disposition | **PORT** the gated-activation repair. **BLOCKED** on the local output reduction; see D2 below. |

## Decision checkpoints for the user

### D1 — Scope of the opt-in DP placement strategy

`ForwardSyncState.get_step_id(batch)` is not a valid report-order key for every configuration the candidate's strategy accepts: dense models and, until W3 lands, MoE with DP>1. The options are to supply a correct Replica-scoped step identity as part of W3, or to narrow the strategy's advertised capability so that it rejects the configurations where no valid identity exists. Recommendation: land W3 first, then re-evaluate whether W3's shared forward identity is itself the correct key; narrow the capability only if it is not. This is recorded rather than blocking, because W2 and W3 proceed independently.

### D2 — Measurement scope of the corrected legacy MoE profiling

Adding the local top-k output reduction changes what the `moe_grouped_gemm` measurement contains. On main that label **already means two different things**: the legacy path measures GEMM1, slice and GEMM2, while the functional path (`:837-869`) measures vLLM's `fused_experts` end to end, which already includes both the activation and `moe_sum`. Adding both to the legacy path converges the two, but no existing column can separate old rows from new: `moe_grouped_gemm_backend` (`frontier/profiling/moe/moe_wrapper.py:50-59`) encodes only `{frontier_loop, vllm_fused, vllm_aiter_mxfp4}` and both vLLM paths emit `vllm_fused`; `measurement_type` records the timer, `quant_signature` the quantization, `model_architecture_profile` the model, and `typed_operator_contracts` the family and TP/EP semantics. The vLLM API version is never written to the CSV.

The candidate's own donor proposal explicitly declined to include `moe_sum`, calling it an operator-ownership decision that must not be folded in silently, and `MOE_FAMILY` (`frontier/operators/families.py:77-133`) and `MoETime` (`frontier/entities/time_components.py:503-547`) still have no reduction term. The candidate's numerical parity test compares against `fused_experts`, whose return is already reduced, so that test presupposes the decision.

The choice is therefore: (a) add a narrowly scoped profiling metadata column that records the measured entry point and scope, then include the reduction; (b) include only the gated-activation repair in this PR and defer the reduction; or (c) defer the whole W6 package. Recommendation: (a), because the two-meanings problem exists on main today independently of this change, and a scope column fixes it once.

## Candidate changes dropped as experiment scaffolding

| Path pattern | Count | Reason |
| --- | --- | --- |
| `tests/e2e/issue26_*` | 52 | Calibration experiment drivers and GPU worker shell scripts. |
| `tests/integration/issue26_*` | 5 | Root-cause-analysis drivers. `issue26_dp_coordinator_reference.py` is reviewed separately before W4. |
| `tests/performance/issue26_*` | 7 | Profiling and microbenchmark drivers. |
| `task_memory/task_2026-09-07_issue26_ttft_h200/` | ~1006 files | Main deliberately removed local task memory in `26b490a`. Read for evidence; not vendored. |
| Whole-file take of `frontier/profiling/moe/moe_vllm_kernel.py` | 1 | Would revert 419 lines of main-only work. |
| `frontier/config/cluster_scheduler_config` import block as written | 1 | The module does not exist on main; the refactor PR creates one with a different content boundary. |
| Non-routing `config.py` hunks (GDN guards, `gdn_input_file`) | several | Unrelated to any work package here. |
| `rtol=0, atol=0` bitwise parity fixture and its `SimpleNamespace` stub | 2 | The stub depends on a `getattr(self, "_cluster_type", None)` form main has since hardened. |

Retained from the candidate's tests: `tests/unit/test_cluster_scheduler_dp_lanes.py`, `test_monolithic_mixed_forward_sync.py`, `test_vllm_dp_load_balancer.py`, `test_moe_routing_runtime.py`, `test_moe_routing_runtime_model_sharing.py`, `test_moe_fused_expert_numerical_parity.py`, `test_collective_sim_zero_payload.py`, each subject to review before adoption.

## Defect found in the candidate itself

The candidate deletes `VLLMv1EngineReplicaScheduler._get_num_waiting_reqs_for_decision_log` while `frontier/scheduler/replica_scheduler/sglang_style_replica_scheduler.py:65` still calls it, and `SGLangStyleReplicaScheduler` subclasses `VLLMv1EngineReplicaScheduler`. Verified: at the candidate revision the caller exists and no definition does, so the SGLang decision-log path raises `AttributeError`. This PR does not reproduce that deletion.

## Open items

1. Whether `_schedule_batch_mode`'s per-replica grouped return order is load-bearing for the consumers of `ClusterScheduleEvent`'s request mapping (W2).
2. RESOLVED. Whether W3's shared forward identity is the correct report-order key for W4, or whether a separate Replica-scoped step identity is required (D1). W3 delivered one step id per monolithic cohort in a single `"forward"` namespace regardless of its lanes' phases, and W4 (`10dd474`) measured that identity at the report boundary: a MoE Replica's keys are non-decreasing with every equal-key pair carrying two distinct lanes, while a dense Replica keeps a per-lane creation counter whose keys interleave under staggered online arrivals. No separate identity was introduced; the dense multi-lane shape is rejected in the constructor instead.
3. CLOSED 2026-09-22. The naming question was first answered (`moe_routing_runtime_path`, a `ReplicaConfig` field), then made moot: the user decided W5 is not ported at all, because the collision it fixes is unreachable while the routing distribution has a single global field, and the override that would make it reachable was judged not worth its configuration surface. See the W5 "Final disposition" row. No public name exists; the CSV column `routing_runtime_path`, the trainer flag `--routing_runtime_path` and main's internal spellings are unchanged.
4. Pipeline-parallel behavior of the component ledgers is untested in both trees.
5. A multi-lane monolithic MoE Replica has no released wrapper: the public MoE examples enforce `ATTN_TP == MOE_TP * MOE_EP` while the runtime enforces `attn_tp * attn_dp == moe_tp * moe_ep`, and those have no common solution above one lane. Its evidence therefore has to come from a direct-construction fixture, never from the fidelity matrix. Whether the release should offer such a wrapper is a product question, not a correctness one, and is left open.
6. The upstream `fused_moe.py` fork change passes a fifth `renormalize` argument to `torch.ops._moe_C.topk_softmax` while the in-tree schema still declares four; the prebuilt extension could not be inspected on this host. Numerically a no-op, but it would raise rather than degrade. Relevant only if W6 native validation runs against the fork's compiled package.

## Final code-review findings

Pending Step 8.

## Maintainer decisions and review dispositions (2026-09-22)

Source: `.local-draft/Frontier_PR34_PR35_Review_and_D1_D2_Decisions_2026-09-22.md`,
maintainer review of PR #34 at `5ef96b5` and PR #35 at `33f0d5a`. Every line
reference below was checked against the tree named in the row.

### D1 — RESOLVED: W3 first, then validate the identity at the report boundary

Approved as written. Order: implement and validate W3; inspect the identity at
the **load-report emission boundary**; reuse it only where its equality and
ordering hold; reject unsupported configurations explicitly rather than adding a
second counter to broaden W4. Dense DP>1 is not admitted by W3 landing. DP1 is a
tested degenerate case that does not validate cross-lane behavior.

Source-backed refinements, verified against `.real-engine/vLLM-BS` at `ea95f571`:

1. **Why per-lane counters group correctly in the reference.** `step_counter`
   is owned by `DPEngineCoreProc` (`core.py:997`, `:1012`), so the maintainer's
   correction stands: per-lane is not itself the defect. What keeps the per-lane
   counters equal across engines is that **every forward, real or dummy,
   performs a DP-group all-reduce** to exchange `num_tokens_across_dp`
   (`forward_context.py:72-84`). That collective, not the every-32-step
   all-reduce in `_has_global_unfinished_reqs` (`core.py:1134-1135`, which only
   decides wave termination), is the lockstep, and it applies to dense models as
   well. A Frontier per-lane counter has no equivalent per-forward
   synchronization for dense DP lanes, so it cannot inherit the property. The
   conclusion (exclude dense DP>1) is unchanged; this is the reason.
2. **What the counter counts and where the key is read.** The increment at
   `:1134` runs in busy-loop step 3, *after* `_maybe_publish_request_counts()`
   at `:1099`. A published key is therefore the count as of the end of the
   previous iteration. Iterations in which every engine is idle `continue`
   before step 3 (`:1103-1105`) and do not increment; iterations that ran a
   dummy forward do. So it counts forwards including dummy forwards since the
   last wave reset (`:1129`), not completed real-batch forwards. This is the
   precise reason §2.3's "literal equality with vLLM's counters is unnecessary
   if grouping and order are preserved" is right, and why Frontier must not try
   to make the numbers match.

**Addition for W4 implementation.** The coordinator keeps one shared
`(last_stats_wave, last_stats_step)` pair across all engines
(`coordinator.py:156-157`). A strictly newer key advances the pair and, when
unpublished changes exist (`stats_changed`), first preserves the prior counts
as a snapshot (`:296-300`); an **equal** key takes neither branch, which is the
expected path for peer engines reporting the same forward; an out-of-order key
produces a **warning only** (`:301-307`) and the counts are still applied
unconditionally (`:308-310`). W4 must not add a hard runtime assertion on
report order that the reference does not have. Key equality per shared forward
is a test invariant, not a runtime abort condition.

### D2 — RESOLVED: include the local reduction, behind a narrow versioned scope identifier

Approved as written, including the compatibility and cache policy in §3.5 and the
validation set in §3.6.

The maintainer's §3.3 correction is confirmed in this tree:

| Claim | Verification |
| --- | --- |
| Alignment is outside the legacy timed region | `moe_align_block_size(...)` at `frontier/profiling/moe/moe_vllm_kernel.py:897`; the legacy `_step` is defined at `:920`. Confirmed outside. |
| The functional entry point aligns internally | The functional `_step` at `:837` calls `_run_functional_fused_experts_iteration`; the `_step` at `:820` is the MXFP4 branch. vLLM 0.10.2 aligns inside `fused_experts`. To be re-verified against the exact supported version before admitting measurements. |
| Double counting is structurally present | Shuffling and grouped GEMM are separate additive terms in **both** accounting paths: legacy `MoETime.total_time()` sums `moe_shuffling_time` and `moe_grouped_gemm_time` (`time_components.py:505`, `:508`, `:531`, `:534`), and the typed path computes `shuffling_time` and `grouped_gemm_time` separately and adds them (`moe_operator_times.py:129-143`). Whether the shuffling predictor is actually populated for functional-backend datasets is **not verified here**; W6 must check it before claiming or denying a live double count. |

Consequence adopted: adding gated SiLU and `moe_sum` does not make the legacy and
functional scopes equal, and no record may claim that it does.

### Disposition of the review comments

| Comment | Verdict | Verification |
| --- | --- | --- |
| R34-01 false success | **ACCEPT, P1** | Both mechanisms reproduced. `baseline_failures` is absent from the `failed` predicate (`run_matrix.py:535`); `complete` (`:437`) tests case-ID presence only; `incomplete` (`:527`) derives from it. All cases failing on both sides yields `compared == 0` and exit 0, which `measure_commit.py:139` prints as `VERDICT: IDENTICAL`. Second path: `list_artifacts` returns `[]` for a missing directory (`compare.py:83-84`), so two absent directories compare equal. |
| R34-02 provenance | **ACCEPT, P1, with one refinement** | Merge-and-overwrite confirmed (`run_matrix.py:294-308`; manifest rebuilt at `:314-327`). `measure_commit.py:82-90` reuses a checkout after checking `HEAD` only. Refinement: the manifest already records `git_dirty_paths` and `cases_executed_in_last_run` (`:319`, `:323`); what is missing is per-case provenance and any *check* of those fields, so the fix is a stamp plus a guard. The partial-run entry points that the guard must cover are `--case-filter`, `--start` and `--limit` on both drivers. Two recorded instances exist in the scratch root; see "Found on re-review". |
| R34-03 evidence record | **ACCEPT, P1; the document's cheaper remedy is not available** | Confirmed: `cases.py` yields 71 cases and the string "71" appears in no tracked document on the refactor branch; `progress.md` still reads "Current step: Step 0", Step 2 `IN_PROGRESS`, Step 7 `NOT_STARTED`. `db15e64..5ef96b5` is one commit touching only `cases.py`, so the production tree is unchanged at the tip, but `candidate_db15e64` holds 67 records and no DP case, and the **baseline label is itself an assembled partial run** (below). Remedy: recapture **both** sides as single clean full 71-case runs after the R34-01/R34-02 fixes land, on the post-fix tip, asserting that its `frontier/` tree equals `5ef96b5`. |
| R34-04 retained checks | **ACCEPT, P2** | PR #34 adds only the four harness files; the seven touched unit files are import-path and monkeypatch-target retargeting to the new modules (37 insertions, 23 deletions). No committed test covers the CLI flag set, the public re-exports, the mixin MRO, or loading a baseline-produced estimator cache. |
| R34-05 bounded split | **ACCEPT, P3** | Non-blocking guidance. |
| R35-01 W2 tests | **ACCEPT, P2** | `test_replica_identity_contract.py:25-34` selects lines beginning `dp_id = ` and asserts `endswith("% self._replica_dp_size")`: a source-string check, not a behavioral placement test. |
| R35-02 wrapper limit | **ACCEPT, P2** | The `cases.py` docstring is accurate as written (it scopes the limit to shipped recipes), but its remedy ("validated by unit tests") is too weak. Corrected remedy: a direct-construction integration fixture driving the real event loop, with deterministic durations injected only at the predictor boundary. |
| R35-03 SGLang consumers | **ACCEPT, P1** | Definition at `vllm_v1_iteration_policy.py:527`; callers at `:573` and `sglang_style_replica_scheduler.py:65`. |
| R35-04 W5 scope | **ACCEPT, P2** | Consistent with the existing W5 audit rows. |
| R35-05 gates | **ACCEPT, P1** | This section is that tracked decision log. |

### Found on re-review

Facts read from the manifests under
`/data/ycfeng/tmp/issue26-correctness-pr/refactor-fidelity/` on 2026-09-22.

| Label | `git_head` | dirty | `case_filter` | executed in last run | `case_count` / lines | `clean_cache` |
| --- | --- | --- | --- | --- | --- | --- |
| `baseline` | `1f694f7` | clean | `dp_` | 4 | **72 / 71** | **False** |
| `candidate_db15e64` | `db15e64` | clean | none | 67 | 67 / 67 | True |
| `candidate_6ab521d` | `6ab521d` | clean | none | 71 | 71 / 71 | True |
| `candidate` | `99922d2` | **3 tracked `frontier/` files modified, 7 untracked new scheduler modules** | none | 67 | 67 / 67 | True |

1. **The baseline is an assembled label.** Its last run was a filtered `dp_`
   run of four cases merged onto the earlier 67 without a cache clean. Every
   per-case artifact is genuine and its source is clean, so the per-case
   equality verdicts stand, but the label as a whole is exactly the R34-02
   pattern, its cache listing is not a clean full-matrix population, and its
   `case_count` (72) disagrees with its results file (71): the merge retained a
   record for a case id that no longer exists in the table. Both the refactor
   comparisons and the W2 measurement in `validation.md` were made against this
   label. Their per-case conclusions are not withdrawn; the "full matrix" and
   cache-name claims must be re-established against a clean baseline.
2. **The contaminated `candidate` label.** Its manifest recorded `git_head`
   `99922d2` against the shared `oversized-module-split` worktree with this
   working tree state, which is the concurrent-edit collision in full:

   - `M frontier/config/cluster_config.py`
   - `M frontier/config/replica_config.py`
   - `M frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py`
   - `M tests/e2e/refactor_fidelity/run_matrix.py`
   - `?? frontier/scheduler/replica_scheduler/vllm_v1_decision_log.py`
   - `?? frontier/scheduler/replica_scheduler/vllm_v1_decode_attn_cohort.py`
   - `?? frontier/scheduler/replica_scheduler/vllm_v1_iteration_policy.py`
   - `?? frontier/scheduler/replica_scheduler/vllm_v1_kv_allocation.py`
   - `?? frontier/scheduler/replica_scheduler/vllm_v1_mtp_wait.py`
   - `?? frontier/scheduler/replica_scheduler/vllm_v1_prefix_cache.py`
   - `?? frontier/scheduler/replica_scheduler/vllm_v1_role_schedules.py`

   The label was deleted on 2026-09-22 with the maintainer's authorization,
   after this list was transcribed here. It must never be used as a
   comparison side.
3. **The W2 record is a mixed-harness measurement.** `candidate_6ab521d` ran the
   `6ab521d` source under the refactor tip's case table and comparator. This was
   disclosed, but the record does not state the harness revision as a field.
   After Checkpoint C rebases #35 onto the fixed harness, W2 should be
   re-measured with harness and source at one revision.
4. **Ownership for the checkpoints.** Checkpoints A and B touch only
   `tests/e2e/refactor_fidelity/`, new tests and task records on the refactor
   branch, which this session authored. Checkpoint C touches the correctness
   worktree that the W3 owner also uses and must be coordinated before it
   starts.
5. **PR #34 status.** It was marked ready for review earlier on 2026-09-22 at
   the maintainer's instruction; the review that followed requests changes
   with three P1 items. Whether it returns to draft until A and B close is the
   maintainer's call.

## Remediation record

One row per accepted comment, with the artifact that closes it. A row is only
marked closed when the evidence for it is committed, not when the change is.

| Comment | State | Closed by |
| --- | --- | --- |
| R34-01 false success | **CLOSED** | `tests/e2e/refactor_fidelity/{run_matrix,compare}.py`; `tests/unit/test_refactor_fidelity_gate.py` (22 tests). Before/after reproduction against the pre-fix harness: four scenarios returned exit 0 and now fail; both controls unchanged. `test_report_2026-09-22_checkpoint_a_fidelity_gate.md` section 3. |
| R34-02 provenance | **CLOSED** | Per-case `source_revision` / `source_dirty` / `harness_revision`; `check_retained_records` refuses a conflicting continuation before running anything; `case_count` counts written lines; cache names compared only for clean unfiltered full runs; `measure_commit.reuse_blocked_reason` reports and refuses a dirty checkout rather than cleaning it. |
| R34-03 evidence record | **CLOSED** | Both sides recaptured as single clean full 71-case runs, one harness revision: 71 of 71 compared, 71 identical, no failures, no missing evidence, no provenance findings, 0 cache differences. `test_report_2026-09-22_checkpoint_b_final_evidence.md`. Status header, `summary.md` and the PR #34 description synchronized. |
| R34-04 retained checks | **CLOSED** | `tests/unit/test_module_split_boundaries.py` (13 tests). Beyond the ask: all 142 baseline-produced pickled estimators load under the split code and 86 predict, which cache-name equality could not show. One pre-existing `NameError` on `ClusterConfig` annotations is pinned, not fixed, and verified to fail identically on `1f694f7`. |
| R34-05 bounded split | **CLOSED** | Guidance applied while writing the new tests; an unused import removed. |
| R35-01 W2 tests | **CLOSED** | `tests/unit/test_cluster_scheduler_dp_lanes.py`: three topologies with the full rotation written out by hand past its wraparound, driven through the public `schedule()`, each run for both MONOLITHIC and PREFILL. The source-text guard is kept with a docstring stating it is governance only. Negative control against the pre-fix method: 12 of 22 fail, controls pass. |
| R35-02 wrapper limit | **CLOSED** | `tests/integration/test_monolithic_mixed_forward_runtime.py` builds `attn_tp=1, attn_dp=2, moe_tp=1, moe_ep=2` directly, runs the real `Simulator` event loop with real admission, ownership and completion, and injects deterministic durations only by wrapping the predictor. It reaches four mixed-phase cohorts, which no wrapper and no matrix case can. On the pre-fix source the same fixture ends with a non-empty scheduler state, so the fixture is shown to detect the defect it exists for. |
| R35-03 SGLang consumers | OPEN | W4. |
| R35-04 W5 scope | OPEN | W5. |
| R35-05 gates | **CLOSED** | This document. |

### What the W2 negative control showed beyond pass/fail

Re-running the strengthened tests against the pre-fix `_schedule_batch_mode`,
taken verbatim from `6ab521d^`, is more informative than a failure count. In all
three topologies the old code produces the **expected sequence exactly** when the
whole stream arrives in one call, and collapses onto lane 0 only when requests
arrive one at a time:

| Topology | Expected, and pre-fix in one burst | Pre-fix, one request at a time |
| --- | --- | --- |
| replicas `[3, 11]`, 2 lanes | `(3,0) (11,0) (3,1) (11,1)` repeating | `(3,0) (11,0)` repeating — every request on lane 0 |
| replicas `[3, 11, 42]`, 3 lanes | lane advances once per replica cycle | every request on lane 0 |
| replicas `[5, 9]`, 3 lanes | `(5,0) (9,0) (5,1) (9,1) (5,2) (9,2)` repeating | every request on lane 0 |

So the hand-derived expectation agrees with the rotation the code already
performed for a single call. The fix did not introduce a placement policy; it
made incremental arrival reach the placement that batch arrival already had.
That is the strongest available statement that W2 is a bug fix rather than a
behavior change, and it is why the matrix is expected to move only the cases
that enter the scheduler many times.
