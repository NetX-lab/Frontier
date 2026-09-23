# Issue 26 Correctness PR — Review Records

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-23 | Step 9 implementation self-review of `d1a2a06..bacdbb4` against the user's quality gates: findings S9-01..S9-08, no source change required; W9-04 and W9-05 recorded as pre-existing. |
| 2026-09-22 | Second Step 9 plan review at the user's direction (quality gates for core-module changes): findings R9-01..R9-08 recorded with dispositions; plan §18.12, design.md. |
| 2026-09-21 | Created with the pinned source snapshot. |
| 2026-09-21 | Step 1 complete: candidate and vLLM audits landed, dispositions recorded, two decision checkpoints raised. |
| 2026-09-21 | Corrected the W3 and W4 rows: the step-id namespace is not partitioned by sync kind, only the open-step binding table is. Verified against `forward_sync_state.py` at `c18eb2c`. |
| 2026-09-22 | W3 delivered and measured; R35-02 closed; the unreleased multi-lane monolithic MoE shape recorded as an open item. |
| 2026-09-22 | Recorded the maintainer's PR #34 / PR #35 review: D1 and D2 resolved, ten review comments dispositioned, each verified against source. |
| 2026-09-22 | Self-review of that record: corrected the lockstep mechanism and counter semantics under D1, the line references under R34-01, the R34-03 remedy (the baseline label is itself an assembled partial run), and added the omissions listed under "Found on re-review". |
| 2026-09-22 | W6 arithmetic delivered with CPU validation; two deviations from the recorded adaptations justified; artifact identity raised as an open decision; native GPU validation NOT_RUN. |
| 2026-09-22 | W6 artifact identity decided as document-only; native parity test added and submitted to an H800 worker. |
| 2026-09-22 | W6 native parity PASS: 8 of 8 at `rtol=0, atol=0` on H800. W7 authorized and delivered; its blocker row closed and the delivery recorded. |
| 2026-09-22 | W7 facts re-verified: the candidate gitlink is unpublished, the three payload defects are confirmed by execution against the published backend, and the fix needs companion-repository authorization. |
| 2026-09-22 | W5 closed without source changes. Corrected the Mechanism A premise: the routing distribution has no per-role override on main, so both W5 mechanisms are unreachable from any released configuration. The user chose to keep the single global field; the drafted implementation was reverted and archived as a patch. |
| 2026-09-22 | Step 8 §14.2 recorded: final self-review of the whole branch diff at `d881357`, with the method stated and one deferred pre-existing defect. |
| 2026-09-22 | External review of PR34/PR35 (`.local-draft/Frontier_PR34_PR35_Current_Code_and_PP_Extension_Review_2026-09-22.md`) verified finding by finding; D2's scope-identifier clause marked SUPERSEDED; disposition table added at the end. |

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
| Delivered 2026-09-22 | Both omitted steps restored in `_run_fused_moe_iteration`, with the two buffers the reference uses, and `moe_grouped_gemm` redefined as the complete local expert computation. 7 new CPU tests in `tests/unit/test_moe_fused_expert_arithmetic.py`, all passing; regression against `HEAD` shows the same single environment-dependent failure on both sides. Full record in `test_report_2026-09-22_w6_fused_expert_arithmetic.md`. |
| Deviation 1 from the adaptations above | The `_custom_ops` import went **inside** the low-level `try`, not outside it. Reason: `fused_moe.py` itself imports `_custom_ops` at its top, so the import cannot fail where the low-level API succeeds, and it therefore cannot perturb the two-branch detection. Placing it there additionally means a build lacking it selects the functional path rather than running an incomplete computation. |
| Deviation 2 from the adaptations above | Main's `SiluAndMul` wrapper was **not** reused. Reason: its `forward` calls `torch.empty` on every invocation (`activation.py:29-33`), which would put an allocation inside the timed profiling step and change what `moe_grouped_gemm` measures. The repair calls `torch.ops._C.silu_and_mul` into a preallocated buffer, exactly as vLLM's `fused_experts_impl` does, and a test asserts one workspace is shared across every profiled step. |
| Measurement ownership, decided | `moe_grouped_gemm` = GEMM1, gated activation, optional activation quantization, GEMM2, local top-k reduction. This is already what the functional backend measured (`fused_experts` returns reduced hidden states), so the repair removes a scope disagreement between the two backends. `MOE_FAMILY` (`frontier/operators/families.py:77`) has no operator for the reduction, so counting it here counts it exactly once without a fifth operator, a new column or a new trained model. The reference's own `record_function("moe_grouped_gemm")` excludes `moe_sum`; that finer split is diagnostics, and the difference is recorded rather than adopted. |
| Magnitude, estimated before implementing | On `a800/qwen3-a3b-30b-moe` the two missing kernels are an estimated 16.5% of the corrected `moe_grouped_gemm` time at 4096 tokens (6.8% median over all rows, 26.3% max). On the two 64-token h800 datasets, 1.1-1.4%. Analytical estimate at 80% of peak HBM, not a measurement. |
| Decided: artifact identity | **Document the limitation, change no metadata.** User decision, 2026-09-22: 不改 metadata，只记录限制. Evidence behind it is unchanged: `resolve_grouped_gemm_backend` (`moe_wrapper.py:50-59`) returns `vllm_fused` for both the low-level and the functional vLLM path, so no column separates an incomplete legacy row from a complete one; `profiling_patch_tag` carries three historical free-text values in `a800/qwen3-a3b-30b-moe/moe.csv` but nothing in the source writes it; nothing in `frontier/` reads `moe_grouped_gemm_backend`, and only `tests/unit/test_moe_native_admission.py:93` asserts the mxfp4 label. Recorded in `docs/profiling/README.md` under the MoE producer: what `moe_grouped_gemm` measures, the size of the pre-fix gap, and that the identity columns cannot date a row, so the remedy is to re-profile rather than infer. |
| Native validation | Test added: `tests/integration/test_moe_fused_expert_numerical_parity.py`, 8 cases, skipping unless CUDA is present and `VLLM_API_VERSION == "0.10.x"`. It drives `_run_fused_moe_iteration` with the buffer shapes, kernel config and alignment `profile_fused_moe_kernel` uses and compares the output tensor against `fused_experts` at `rtol=0, atol=0`. Neither Torch environment on this host selects the repaired path (vLLM 0.11.0 and 0.28.0), so it runs on an H800 worker under the official `vllm/vllm-openai:v0.10.2` image. Result recorded in `test_report_2026-09-22_w6_fused_expert_arithmetic.md`. |
| Measured scope today | The whole `_step` body, timed either with per-iteration CUDA events and a synchronize (`:603-626`) or through `record_function("vidur_moe_grouped_gemm")` (`:629-653`). All buffers are allocated outside the timed region. |
| Disposition | **PORT** the gated-activation repair. **BLOCKED** on the local output reduction; see D2 below. |

### W7 — Optional zero-payload collective-sim backend

Facts re-verified 2026-09-22, superseding the specification-time record in `plan.md` A7.

| Item | Finding |
| --- | --- |
| Repository access | Not a problem. `fwyc0573/frontier-htsim` is public and readable, `pushed_at` 2026-06-08, one branch `main` at `b8518afcc310f0fe0e3ce52ba6b4f0bf57a3be04`, which is exactly the gitlink this branch and main already pin. |
| Candidate commit | **Unpublished, not inaccessible.** `e564935d3874d8c71b52a554ab7c9a72e5e19f68` returns HTTP 422 `No commit found for SHA`. No local object store on this host contains it: the submodule directory is empty in every checkout, and `git cat-file -t` fails in the main repository. The candidate's own branch still records it as its gitlink. |
| Where the fix lives | Entirely in the companion repository. The donor test drives `collective_sim_core.predictor.predict_collective_time` and `htsim_runner.py`, both inside the submodule. Frontier's side of this package is the gitlink and nothing else. |
| Defect 1, confirmed by execution | An explicit zero payload is rejected as a missing field. `htsim_runner.py:2349` tests `getattr(args, k) in (None, "", 0)` over a required-field list that includes `tensor_bytes`. Running the published `main` runner with `tensor_bytes = 0` and with the field deleted produces the identical `exit=2, Error: missing required fields: ['tensor_bytes']`. |
| Defect 2, confirmed by execution | A negative payload is accepted. `--tensor-bytes` is a bare `type=int` with no lower bound and no schema check, so `-1` passes validation and reaches the simulator invocation. |
| Defect 3, confirmed by execution | An explicit CLI zero loses to a positive spec value. `set_if_none_or_zero` (`htsim_runner.py:1398-1406`) overwrites the CLI value when it is `0`, so `--tensor-bytes 0` against a spec of 32768 yields 32768. |
| Reachability from Frontier | Real. `base_cc_backend._validate_data_size` rejects only negative sizes, so Frontier passes zero through. `moe_operator_times.py:512` computes `data_size_bytes = embedding_dim * 2 * routed_tokens` and hands it to `predict_all_to_all`; an EP lane with no routed tokens in a step makes that zero. `predict_reduce_scatter` additionally floor-divides by the device count. A MoE EP run under `--cc_backend_config_type collective_sim` therefore aborts on a legitimate empty collective. |
| Why Frontier cannot fix it alone | A zero-byte collective is not a zero-cost collective. The donor test asserts the intra-server latency term survives at payload 0 (`7 x 0.5 us`, `network_ms == 0`), which is also the plan's requirement. Short-circuiting to `0.0` in Frontier would change the backend's synchronization semantics rather than accept the input. |
| Cost of the fix | Three small edits in the companion repository: drop `tensor_bytes` from the zero-means-missing list while keeping it required, add a `>= 0` check, and make the CLI precedence distinguish "unset" from "explicitly zero". No Frontier source change; Frontier moves its gitlink and gains the donor's CPU test. |
| Blocker | **CLEARED 2026-09-22.** The user authorized the companion-repository option ("1.授权"). Publication order as proposed: the backend commit first, Frontier's gitlink after. |

### W7 delivery

| Item | Outcome |
| --- | --- |
| Companion commit | `fwyc0573/frontier-htsim` branch `fix/zero-payload-input-handling`, `eb7bc4f`, companion **draft** PR 1. Three source edits plus a narrowed `.gitignore`; no change to flow generation, topology modelling or latency arithmetic. |
| Design choice worth flagging | The required-field check became a table of `(field, zero_is_valid)` rather than a special case for `tensor_bytes`, per the AGENTS.md rule that a growing category gets a table. A first draft added a fourth merge helper; it was collapsed into the existing `set_if_none_or_empty`, whose semantics are already exactly right for this field, rather than left as a near-duplicate. |
| Frontier change | Gitlink `b8518af` -> `eb7bc4f` and one new test module. No Frontier source file changed, which matches the candidate branch: its own `collective_sim_cc_backend.py` is byte-identical to this branch's. |
| Frontier test scope | The donor test re-drove the runner CLI and the submodule predictor. That is now the companion repository's own coverage, so the Frontier module tests the Frontier boundary instead: `CollectiveSimCCBackend.predict_all_to_all` and `predict_reduce_scatter` on the canonical `TP=4 x DP=2, EP=8` pod. |
| Defect found while validating | The gitlink bump made three repository-governance scans read 38 vendored files as Frontier's own, and one of them stopped parsing. Repaired with `tests/frontier_sources.iter_frontier_sources()`. Pre-existing and reachable by anyone who initializes the optional submodule; not caused by the backend fix. |
| Carried forward | Frontier's gitlink points at a commit on an unmerged companion branch. `git submodule update --init`, the documented command, resolves it; `git submodule update --remote` would follow `.gitmodules`' `branch = main` and drop the fix. Re-point the gitlink at `main` once companion PR 1 merges. |

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

### D2 — RESOLVED: include the local reduction (the scope-identifier clause is SUPERSEDED)

**Superseded in part, 2026-09-22.** The "narrow versioned scope identifier" and
its §3.5 compatibility/cache policy were overtaken by the user's later decision,
recorded in the W6 table above under "Decided: artifact identity": 不改
metadata，只记录限制 — document the limitation, change no metadata. The
inclusion of the local reduction, the §3.3 verification table and the
"Consequence adopted" paragraph below remain in force. Only one rule is active:
the documentation-only decision (external review C35-05).

Approved as written at the time, including the compatibility and cache policy in
§3.5 and the validation set in §3.6.

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

## Step 8 §14.2 — Final review of the complete diff against the PR base

| Field | Value |
| --- | --- |
| Reviewed revision | `d881357` |
| Base compared against | `refactor/oversized-module-split` @ `6ef0a3c` (32 commits) |
| Method | **Self-review.** One reviewer, the same agent that wrote the change, reading `git diff 6ef0a3c..d881357` restricted to `frontier/ tests/ docs/ examples/` — 1925 lines — line by line with surrounding context opened where a hunk did not stand alone. This is not independent review; no second party inspected the diff. |
| Task records excluded from the line-by-line pass | `task_memory/` and `AGENTS.md`, reviewed separately as documentation. |

### Answers to the §14.2 questions

| Question | Finding |
| --- | --- |
| Is every change linked to a demonstrated defect, an essential regression test, or a directly related simplification? | Yes. Each of W2, W3, W4, W6, W7 carries a negative control that fails against the unrepaired source, recorded in `validation.md`. W5 produced no source change. The only additions without a paired defect are `tests/frontier_sources.py`, which exists because the gitlink bump exposed a real scan defect, and the `AGENTS.md` scheduler list, which documents `VllmLoadBalancingClusterScheduler` added by W4. |
| Are state ownership and initialization explicit? Are repeated fallback checks or parallel state representations still present without a reason? | W3's waiting rooms are owned by one structure keyed by step id, initialized at creation and deleted at release; the tests assert no dangling room survives a cohort. No fallback chain was added. |
| Do source batches keep their own shape and progress? Are terminal events and ownership transitions unique? | Yes; this is what the W3 mixed-source tests and the four deliberate-defect controls check, including a control that deliberately merges two sources' progress and one that emits a duplicate terminal event. |
| Are config, layer, routing-runtime, precision, and measurement identities preserved through both fresh and cached paths? | Yes. The §14.1 cold/warm pair is direct evidence: the freshly trained predictors and the persisted ones produce byte-identical `request_metrics.csv`. |
| Is any operation omitted, counted twice, or relabeled without compatible metadata? | The one relabelling risk is W6's `resolve_grouped_gemm_backend`, which labels both vLLM paths `vllm_fused`. The user decided not to change the metadata; the limitation is written into `docs/profiling/README.md` instead. |
| Did the patch preserve current-main model/backend support and demand-driven reporting? | Yes. The unit `FAILED` set is identical to the baseline, all 16 architecture examples pass, and the stage-reporting path is unchanged. |
| Are tests checking production behavior rather than copying the implementation or replacing the behavior under test with a stub? | The W6 CPU tests replace the native calls with plain-Torch references and say so in the file; their authority is the native GPU parity run, which uses the real kernels. Every other new test drives the production object. |
| Are any workstation paths, credentials, datasets, weights, generated traces, caches, or unreachable submodule references staged? | No. The branch diff over `frontier/ tests/ docs/ examples/` was scanned for `/data/ycfeng`, `/home/brainpp`, `BRAINPP_`, `ACCESS_KEY`, `SECRET`, `password`, `token=` with no hit. The gitlink resolves from the published remote, proved by the clean-checkout run in the W7 report. |
| Can a reader understand the names and functions without the historical calibration conversation? | The names follow the surrounding ML-system vocabulary. `VllmLoadBalancingClusterScheduler`'s constructor rejects every configuration outside its narrow support, and `AGENTS.md` now states that no placement or timing equivalence with a real vLLM deployment is claimed. |

### Cleanup performed in this pass

| Item | Action |
| --- | --- |
| `set_if_unset` in the companion runner | Removed. It was a near-copy of `set_if_none_or_empty`, and the `--tensor-bytes` argparse default of `None` makes the two behave identically. |
| `pipeline_time` in `prefill_collective.py` | Verified the removal is safe: the value now comes from `final_timing.pipeline_time` at line 232. |
| `outputs/examples/` generated trees | Removed after confirming 0 tracked files there; the 110 tracked files under `outputs/` are untouched. |

No speculative abstraction was added in the cleanup pass. The affected tests were
re-run afterwards, with results in `validation.md`.

### Pre-existing defect found, not repaired

`AGENTS.md` §Tests points at `tests/debug/` and `comm_backend_tests/`, neither of
which exists here or on `origin/main`; the same missing tree causes 10 of the 84
baseline unit failures. Reproduced on the base, reported as a baseline failure,
and recorded in `future.md` rather than repaired, because the correct fix is a
decision about the published test surface and is unrelated to Issue 26.

## External review 2026-09-22 — findings disposition

Reviewed revisions: PR34 `6ef0a3c`, PR35 `0137269`. Each finding was checked against source before any edit. Statuses: `FIXED` (code or record changed and verified), `ACCEPTED_LIMITATION` (true, recorded, not changed), `SUPERSEDED` (overtaken by a dated decision), `OPEN` (still to do). Commit SHAs are in the branch log; the evidence file is `test_report_2026-09-22_review_corrections.md` unless stated.

| Finding | Verified as | Status | Where / evidence |
| --- | --- | --- | --- |
| C34-01 cache comparison eligibility ignores `cases_executed_in_last_run` | Confirmed: `compare_labels` read only `cache_clean_before_run` and `case_filter`; `--start`/`--limit` leave no filter | FIXED | PR34 `2310417` (runner + 4 gate tests), merged as `0d025f8`; Checkpoint B verdict re-derived, unchanged (`task_2026-09-21_oversized_module_split/test_report_2026-09-22_cache_eligibility_correction.md`) |
| C35-01 decoding request in a prefill-mode mixed batch uncredited at a dense layer | Confirmed by call path: `complete_dense_layer(phase="prefill")` → `handle_prefill_sync_collective`, which credits nothing; only the shared forward and decode helpers credit | FIXED | `advance_decode_layer` helper (validate then increment) used by all three completion paths; dense prefill-mode source credits its decoding members. Unit: mixed source at a dense layer +1 for the decoder, 0 for the prefiller, pure-prefill control credits nothing; `MoE -> dense -> MoE` credits 1, 2, 3. Real loop: hybrid `moe_layers_enum="0,2,3"`, 4 mixed dense completions, 10 decode tokens all peaking at 4 layers; the pre-fix source peaks 4 of them at 3 |
| C35-02 W6 report claims legacy scope equals functional scope | Confirmed: the functional entry aligns inside `fused_experts` (vLLM 0.10.2 `fused_moe.py:1718`), the legacy path aligns before `_step`; shuffling and grouped GEMM are additive in both accounting paths | FIXED (records) | W6 report §5 scope table; `docs/profiling/README.md`; `summary.md`; PR35 body. Live double count for functional datasets: not verified, not claimed |
| C35-03 FP8 test omits `block_shape`; "8 of 8 at `rtol=0, atol=0`" overstates | Confirmed: `block_dims` was passed, `block_shape` was not; the FP8 test asserts shape and finiteness only | FIXED (test + wording + native rerun) | `block_shape=block_shape` added; CPU test pins both GEMM invocations receive it (`[128, 64]`) and `None` when omitted; report §8, `summary.md`, PR35 body restated as seven comparisons plus one structural check. Native rerun authorized and executed 2026-09-22: `exp-0922-202645-561899`, 8 passed in 14.27 s, exit 0 (W6 report §8) |
| C35-04 unconditional `import torch` adds a collection error | Confirmed: 11 collection errors in the minimal environment versus 10 on the base | FIXED | `pytest.importorskip("torch")` before importing the profiler module; minimal env: `1 skipped`; torch env: 9 passed; unit suite errors back to 10 |
| C35-05 records inconsistent (D2 metadata rule vs documentation-only; W2-checkpoint diff claim; blanket vLLM-comparison exclusion; PR34 "Draft") | Confirmed on all four points | FIXED (records) | D2 heading marked SUPERSEDED in part with a link to the dated decision; PR35 body scopes the `ceac2b4` diff claim to the W2 checkpoint and amends the exclusion for the authorized scheduler-level comparison; `progress.md` status table current; PR34 body says "open for review" |
| Review's "PR35 mergeable=false" | Stale: GitHub reports `MERGEABLE` for both PRs; PR34 `isDraft=false` | ACCEPTED_LIMITATION (of the review) | `gh pr view` 2026-09-22 |
| P9-01 room-only hook rule; "steady state needs no change" | Accepted: the reference branch is a three-way conjunction; depth-`P` completes `B_(k-P+1)` | FIXED (plan) | `plan.md` §18.3, D9-1, §18.11 state table; `design.md` W9 |
| P9-02 K3 as written breaks peer-key equality | Reproduced on this branch's balancer: equal keys → lane 0; fresh key → partial snapshot `[(0,3),(0,0)]`, lane 1 | FIXED (plan) | K1, K3-as-written, stride rejected as acceptance basis; invariants I1–I6; rule deferred to the design checkpoint after P1 establishes iteration membership |
| P9-03 emission log alone cannot show the placement path | Accepted | FIXED (plan) | Instrumentation chain (iteration, emission, coordinator receive/publish with snapshot id, frontend application, routing), named fields, correlation ids, changed-file list under G1, T2 `SCENARIO_NOT_REACHED` rule |
| P9-04 boundary index is not alignment; unmodified PP2 baseline cannot run | Accepted; constructor rejects PP>1 at `0137269` | FIXED (plan) | CPU reference-loop oracle, causal join, first-cause labels, controls table (rejected production / test-only guard-lifted / corrected / round-robin optional) |
| P9-05 PP3 needs a valid layer count; behavioral matrix | Accepted: tiny Qwen has 8 layers, `_model()` fixture 4, `num_layers % PP == 0` enforced | FIXED (plan) | PP3 CPU fixture with 6 or 12 layers; matrix in §18.11 |
| P9-06 work graph and acceptance language | Accepted | FIXED (plan) | §18.5 graph, §18.1 C1–C5 |
| Package F (W9 implementation) | — | OPEN by instruction | Not started (user: 暂不开启 new subtask) |

## Second plan review 2026-09-22 — user-directed quality gates for Step 9

Reviewer: this session, against `c231322`, at the user's direction ("确保当前计划的代码模块的实现/改动/重构是基于整体codebase的 ... 禁止hard-coding，禁止临时补丁，禁止过度防御，禁止冗余性设计和实现，禁止使用ai味命名函数和变量"). Inspected: `vllm_load_balancing_cluster_scheduler.py`, `vllm_dp_load_balancer.py`, `base_cluster_scheduler.py:417-470`, `base_replica_scheduler.py:36-60, 440-480, 1048-1063`, `forward_sync_state.py`, `global_batch_end_event.py:150-215`, `replica_schedule_event.py:80-175`, `vllm_v1_iteration_policy.py:533-545`, both DP-placement test modules, `AGENTS.md:620`, reference `core.py:318-372, 1075-1137`, `coordinator.py:280-312`. Full text in `plan.md` §18.12.

| Id | Gate | Finding | Disposition |
| --- | --- | --- | --- |
| R9-01 | redundancy / over-design | Readiness classification has no DES counterpart; the room test is real but computed by the policy from the lane's existing `num_running_batches`; only the admission-only row is new; one rule for all PP | Plan amended: hook payload = completion hook signature |
| R9-02 | grounded in codebase | Completion key `get_step_id(batch)` names the scheduling iteration; correct only at PP=1 | D9-2 amended: key both kinds by the observing iteration; C2 verifies PP=1 |
| R9-03 | reuse before inventing | `ForwardSyncState._next_step_id_by_replica` meets I1–I4, I6; I5 gap measured, not assumed | P1 tests it first; decision at the checkpoint |
| R9-04 | over-defense / layering | Constructor-required `_cluster_scheduler`; do not repeat the `getattr`/`hasattr` reach-ups | D9-1 wording; asymmetry recorded in `design.md` |
| R9-05 | redundancy | Oracle must not re-implement the balancer's coordinator/frontend | P1(a) narrowed to the engine loop |
| R9-06 | naming / test surface | Plain names; existing key assertions remain valid under I1–I2 | P3/P4 wording |
| R9-07 | value / size | Frontier change is small and user-requested; validation must not leak into `frontier/` | Boundary stated |
| R9-08 | reference precision | DP engines not iteration-lockstep (all-reduce every 32 steps) | §18.2 row amended; G5 first-cause label |

No source change results from this review; Step 9 execution remains unstarted pending the user's start signal.

## Step 9 implementation self-review 2026-09-23

Reviewer: this session. Reviewed range: `d1a2a06..bacdbb4` (P2 `2ffe78d`, P4 `bacdbb4`). Gates: the user's core-module rule ("任何引入的修改和实现都应该是高价值的 ... 禁止hard-coding，禁止临时补丁，禁止过度防御，禁止冗余性设计和实现，禁止使用ai味命名函数和变量") and the AGENTS.md development gates. Inspected: the four changed `frontier/` files in full diff, the admission loop at `base_replica_scheduler.py:1050-1075`, the completion hook call at `global_batch_end_event.py:180-185`, `base_replica_scheduler.py:51` (stage count), and the test diffs. Evidence of behavior: `validation.md` Step 9 and the W9 report §4–§6.

`frontier/` change: 4 files, +102/−28 lines (`git diff --numstat d1a2a06 bacdbb4 -- frontier/`). `VllmLoadBalancingClusterScheduler` is 158 lines; the largest touched module, `base_cluster_scheduler.py`, is 1,946 lines, under the 2,000-line gate.

| Id | Gate | Finding | Disposition |
| --- | --- | --- | --- |
| S9-01 | value | The change gives the opt-in policy the PP>1 support the user asked for, and adds one fidelity gain: a schedule-time report while the pipeline has room, which the reference engine publishes and the pre-P2 policy could not express. At PP=1 every admission is held, so C2 shows no behavior change (24 of 24). | Accepted |
| S9-02 | redundancy / superseded path | The completion key no longer reads `ForwardSyncState.get_step_id`; the import and the per-layer step-id key are deleted in the same change. No second key path remains. | Accepted |
| S9-03 | reuse before inventing | The key reads the stage-0 forward group that `StageExecutionContext` already binds. The new property `joinable_forward_group_id` exposes the existing `_forward_group_id` / `_forward_group_sealed` / `_next_forward_group_id` state; it adds no state. | Accepted |
| S9-04 | correctness of the single held slot | One `_held_key` per lane suffices. An admission is held only when it fills the pipeline (`running == PP`); the loop admits only while `running < PP` (`base_replica_scheduler.py:1052`); and `GlobalBatchEndEvent` calls `on_batch_end` (decrement) and then the completion hook in the same handler, before any later `ReplicaScheduleEvent` can admit again. So a held key is always consumed before the next held admission on that lane. The unit case "a completion plus the admission it makes room for" pins that order. | Accepted, no guard added |
| S9-05 | over-defense | No new guard, fallback or `hasattr` reach-up. The `_lane_index` type check is the pre-existing one, extracted so both hooks share it. The inert base hook returns `None` and costs one call per MONOLITHIC/PREFILL admission for the other schedulers; the fidelity matrix is 71 of 71 identical. | Accepted |
| S9-06 | hard-coding | No literal enters the path. The pipeline depth comes from `replica_config.num_pipeline_stages`, the same source as the lane's `_num_stages` (`base_replica_scheduler.py:51`). | Accepted |
| S9-07 | naming | `on_replica_batch_scheduled`, `joinable_forward_group_id`, `_held_key`, `_last_admitted_key`, `_next_report_key`: each names the scheduling concept it holds and follows the neighboring `on_replica_batch_end` and forward-group vocabulary. | Accepted |
| S9-08 | test surface | The tests stay out of `frontier/` (R9-07). Each new or changed unit case fails on the pre-P2 tree (19 of 19). The integration test checks each report against the reference iteration kind, not against Frontier's own output, and its discriminating case separates the policy from a completion-reporting control. | Accepted |

Found during validation, both outside this change:

- W9-04, a MoE `attn_dp=4` online deadlock from a stale first-layer placeholder. It is reachable under `round_robin` on this branch, and the pre-Step-9 tree `d1a2a06` stops in the same state, so Step 9 does not cause it. The prototype fix is not applied and awaits the user's decision (`issues.md`).
- W9-05, requests lost mid-decode under KV pressure in `vllm_v1`. It is also present on `origin/main` and is not diagnosed (`issues.md`).

Result: no source change required by this review. G3–G5 remain blocked on GPU authorization.
