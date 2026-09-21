# Issue 26 Correctness PR — Review Records

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-21 | Created with the pinned source snapshot. |
| 2026-09-21 | Step 1 complete: candidate and vLLM audits landed, dispositions recorded, two decision checkpoints raised. |

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
| Defect on main | Present at three layers. `frontier/events/replica_stage_schedule_event.py:157-184` selects the prefill or decode sync path from the lane's own `num_prefill_tokens`; `frontier/scheduler/utils/sync_state.py:29-34` allocates two independent waiting rooms for MONOLITHIC; `frontier/scheduler/utils/forward_sync_state.py:38-41` partitions the step-id namespace into `"prefill"` and `"decode"`. A mixed-phase forward puts one required lane in each room, neither peer is idle, and both rooms stall. |
| Clean ports | The `"forward"` sync kind, and the cross-lane duplicate-request guard in `frontier/scheduler/utils/ep_wave_inputs.py`. |
| Blocked hunk | The candidate's decode final-metrics hunk calls `scheduler._create_corrected_execution_time_for_metrics(...)`, which **main deleted**; `_create_prefill_corrected_execution_time_for_metrics` also changed signature and now lives at `base_cluster_scheduler.py:1138`. Porting verbatim raises `AttributeError`. Main additionally added `metrics_store` and `ep_wave_reporting_enabled` plumbing to `ep_wave_schedule.py` and `prefill_collective.py` that the candidate lacks. |
| Disposition | **ADAPT** for the lifecycle change as one coherent unit; **BLOCKED** for the metrics hunk until it is rewritten against main's current execution-time ownership. |
| Planned test | The behavior matrix in `plan.md` §9, including at least one test that drives the real event loop with overlapping prefill and decode and injects deterministic times only at the predictor boundary. |

### W4 — Opt-in vLLM-style DP placement

| Item | Finding |
| --- | --- |
| New public surface in the candidate | `ClusterSchedulerType.VLLM_LOAD_BALANCING`, the CLI token `--cluster_scheduler_config_type vllm_load_balancing` (no other new flag), `VllmLoadBalancingClusterSchedulerConfig`, a new `frontier/config/cluster_scheduler_config.py`, a registry entry, `RequestLoad`, `VllmDPLoadBalancer`, `BaseClusterScheduler.schedule_at` / `on_replica_batch_end`, and `BaseReplicaScheduler.get_request_load`. |
| Reference semantics confirmed | Engine selection scores `waiting * 4 + running` in the **frontend**, `vllm/v1/engine/core_client.py:1146`. The coordinator transports unweighted `[waiting, running]` pairs. Timing constants confirmed: `min_stats_update_interval_ms = 100` (`coordinator.py:116`, a floor rather than a period), the bare literals `5000` (`coordinator.py:198`) and `50` (`coordinator.py:202`). The specification's grouping of the weight 4 with the coordinator constants is corrected here: it belongs to the frontend. |
| Request populations confirmed | Admitted-but-not-yet-scheduled requests count as **running** (`scheduler.py:812-813`, appended inside `schedule()` before the model runs); preemption moves running to waiting within the same step (`:507`, `:537-538`); requests finishing this step are in neither (`:1381-1385`, before `make_stats` at `:1413`). |
| Report suppression confirmed | Two independent stats producers exist; only the DP one feeds the coordinator, and it is suppressed whenever the count pair is unchanged (`core.py:1081`). That is why the 5 s heartbeat exists and why the step counter is a sparse tag rather than a dense counter. |
| Step-identity problem | The candidate uses `ForwardSyncState.get_step_id(batch)` as the report-order key (`vllm_load_balancing_cluster_scheduler.py:52-56`, latched at `vllm_dp_load_balancer.py:64-67` against a single scalar shared across engines). On main that identity is **invalid for dense models** (`base_replica_scheduler.py:459-466` sets `_forward_cohort_id` from a per-DP-lane creation counter that dense never promotes to a Replica-scoped counter, and the policy's guard does not require MoE), **invalid for MoE with DP>1 until W3 lands** (the kind-partitioned namespace gives the two lanes different ids), and valid for DP1. Idle participants are never reported. The id also advances per layer (`sync_entry.py:64`, `forward_sync_state.py:133-139`), so it is monotonic but is not a vLLM step counter. |
| Disposition | **BLOCKED on W3**, then ADAPT. This makes the execution order W2 (independent) then W3 then W4. |
| Decision raised | See D1 below. |

### W5 — Routing load distribution versus routing implementation identity

| Item | Finding |
| --- | --- |
| Defect on main | Present and reachable from the public CLI. Two independent mechanisms let one routing implementation's cost model be used for another. |
| Mechanism A | Routing distribution is settable per role (`config.py:4242-4247`, `:4257-4273`, `:4302`), so a PDD run can use `PREFILL=balanced` and `DECODE=random`, which resolve to different runtime paths (`moe_routing_runtime.py:29-32`). `trained_model_signatures` is one set shared across clusters (`shared_prediction_model_manager.py:702`, `:707`, `:778`) and `ffn_signature` (`:1377-1382`) carries **no routing term** (re-verified). The second cluster therefore returns early at `:1384-1386`, before `_validate_moe_dataset_contract` at `:1433-1439`, so there is no fail-fast, and it predicts `moe_gating_routing_topk` using the other cluster's estimator through `_models_view_for_family` (`:4358-4388`), which matches on layer identity alone. |
| Mechanism B | When both clusters do train, `_store_model_precision` keys on `(model_name, identity)` at `:4238` with a layer-shape-derived identity, so the second model overwrites the first. |
| Where identity does reach today | Only dataset row selection (`:1294-1326`, `:1235-1265`) and the per-call `moe_df_cache` key (`:1486-1505`), which is scoped to a single cluster call and therefore never prevents the collision. |
| Correction to the specification's premise | The **persistent disk cache is not the hole**. `_get_model_hash` (`:4008-4056`) hashes `df.to_json()` at `:4033`, and `_load_moe_df` filters rows without dropping columns (`:3628`, `:3712-3720`), so cached artifacts are separated incidentally. |
| Disposition | **PORT** the resolver override, the config field and its copy, the predictor helper, dataset admission, the training signature term, and the per-model training identity. **ADAPT** the registry key widening: the candidate's helper rewrite enumerates only `eager` and `kernel_only` and would break main's third `device_event` measurement family (`:4160-4162`, `:544-551`). |
| Two implementation caveats | The candidate appends `_routing{path}` to the whole `ffn_signature`, which separates every MoE FFN model rather than only the routing-topk model. That is correctness-safe but over-broad and should be narrowed. Main's standalone trainer already calls this concept `routing_runtime_path` (`frontier/training/cli.py:153`), so the candidate's `moe_gating_routing_runtime_path` would be a third spelling; one name must be chosen before any public flag exists. |

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
2. Whether W3's shared forward identity is the correct report-order key for W4, or whether a separate Replica-scoped step identity is required (D1).
3. Which of the three existing spellings becomes the single public name for the routing implementation identity (W5).
4. Pipeline-parallel behavior of the component ledgers is untested in both trees.
5. The upstream `fused_moe.py` fork change passes a fifth `renormalize` argument to `torch.ops._moe_C.topk_softmax` while the in-tree schema still declares four; the prebuilt extension could not be inspected on this host. Numerically a no-op, but it would raise rather than degrade. Relevant only if W6 native validation runs against the fork's compiled package.

## Final code-review findings

Pending Step 8.
