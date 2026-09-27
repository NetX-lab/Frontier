# Issue 26 Correctness PR — Summary

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-25 | W9 row: C3 scope stated (balancer on native reports, not Frontier's report emission); the G3b fixes `dfb0b25`, `20f0f94`, `df8ebc6` added. |
| 2026-09-24 | Open items: W3-R5/W3-R6 fixed and merged; F-R5 on stacked draft PR 38. |
| 2026-09-24 | Gitlink re-pointed at the merged companion `main` (`d28fe917`, `1b11eff`); the open item removed. |
| 2026-09-24 | Rerun review G3a lead 26: the W6 row and Limits now cite the native rerun of the current FP8 step. |
| 2026-09-24 | Native W6 FP8 rerun of the current FP8 code PASS (`exp-0924-114241-429126`); follow-up decisions for items 1-7 recorded. |
| 2026-09-24 | W7-R4 fixed by the user's choice of option (a) (companion `e922c77`, gitlink `9adf759`); removed from the proposals; the final-validation worktrees removed. |
| 2026-09-24 | Workflow `wf_7606e14e-f10` results reconciled: the W6 negative-control and CPU-test rows corrected (`6828581` pins the profiler's config and alignment arguments); W7-R4 added to the proposals; the GPU scope in Limits corrected. |
| 2026-09-24 | Fix review after G5 (`c647e95`..`6aee289`) added to Work packages, the validation table and the open items; C4 decided (option a); S43 and S42 moved to their own tasks; W6 and W7 rows corrected. |
| 2026-09-23 | Step 9 G3–G5 results added to W9, the validation table and the open items. C3 PASS, C4 `SCENARIO_NOT_REACHED`, and findings S43/S42 await review. |
| 2026-09-23 | The W9-05 fix (`75c1140`) added to Work packages and Deliverables; its follow-ups replace the W9-05 open item. G3–G5 authorized by the user and in preparation. |
| 2026-09-23 | W9 and the W9-04 fix (`2ffb062`) added to Work packages and Deliverables; open items reduced to G3–G5 and W9-05 (deferred as a separate item by the user). |
| 2026-09-23 | Status only: Step 9 P1–P5 complete on the CPU (`2ffe78d`, `bacdbb4`); D9-2 decided as group-anchored. Open items W9-04, W9-05 and G3–G5 added below. The Step 0–8 archive is otherwise unchanged. |
| 2026-09-23 | Status only: Step 9 in progress. W9-01 fixed via PR 36 and merged forward (composition check PASS); P1 complete; D9-2 proposed, awaiting the user's decision (`progress.md`, plan §18.15). The Step 0–8 archive below is unchanged. |
| 2026-09-22 | FP8 native rerun with the corrected `block_shape` wiring PASS (`exp-0922-202645-561899`, 8 passed). Step 9 plan reviewed a second time against the user's quality gates (plan §18.12); still not started. |
| 2026-09-21 | Placeholder created at Step 0. |
| 2026-09-22 | Completion archive written at Step 8. |
| 2026-09-22 | External review corrections: W3 dense-layer credit for decoding requests in a mixed batch (C35-01); W6 native result restated as seven comparisons plus one FP8 structural check, FP8 `block_shape` wiring corrected (C35-02/03); optional-torch skip (C35-04); records aligned (C35-05). Step 9 remains planned, not started. |

## Overview

Seven correctness candidates were extracted from `bug/ttft-check` by source
audit, re-derived against current `main`, and each one was either implemented
with a negative control that fails on the unrepaired source, or closed with the
reason recorded. Nothing was merged wholesale from the candidate branch.

Delivered as a stacked pair of draft PRs:

| PR | Branch | Base | Role |
| --- | --- | --- | --- |
| [#34](https://github.com/NetX-lab/Frontier/pull/34) | `refactor/oversized-module-split` | `main` @ `1f694f7` | Brings the four modules this work edits under the 2,000-line gate, behavior unchanged |
| [#35](https://github.com/NetX-lab/Frontier/pull/35) | `fix/issue26-correctness-pr` | `refactor/oversized-module-split` @ `6ef0a3c` | The behavior changes; 33 commits, final head `8730509` |
| [fwyc0573/frontier-htsim#1](https://github.com/fwyc0573/frontier-htsim/pull/1) | `fix/zero-payload-input-handling` | `main` @ `b8518af` | Companion backend fix, commits `eb7bc4f` and `ff11ee6` (2026-09-24) |

Latency calibration and any Frontier-versus-vLLM end-to-end comparison were out
of scope throughout. Issue 26 stays open. All three PRs are draft.

## Work packages

| # | Fix | Outcome |
| --- | --- | --- |
| W2 | Round-robin DP placement keeps rotating across scheduling calls | Landed. `6ab521d`, tests strengthened in `ceac2b4`. |
| W3 | A monolithic Replica completes one shared forward across mixed prefill and decode source lanes | Landed. `65ed8a7`. A decoding request inside a prefill-mode mixed batch was still missing its credit at a dense layer; repaired 2026-09-22 (C35-01, `test_report_2026-09-22_review_corrections.md`). |
| W4 | Opt-in vLLM-style DP request placement, off by default and bounded in its constructor | Landed. `10dd474`. |
| W5 | Routing implementation identity separated from expert-load distribution | **Closed, not ported**, by user decision after the premise check showed the collision unreachable from any released configuration. Drafted implementation reverted before commit and archived as `w5_reverted_moe_routing_runtime_path.patch`. |
| W6 | Legacy fused-MoE profiling performs the real gated expert computation | Landed. `7269bac`, native parity test `697f219`, identity limits documented in `79f599a`. |
| W7 | The collective-sim backend accepts an empty collective | Landed companion-side. Frontier gitlink moved in `1b95187`; governance scans narrowed in `beded3c`. |
| W9 | The opt-in vLLM DP placement policy supports pipeline parallelism (Step 9) | Landed on the CPU: `2ffe78d`, tests `bacdbb4`. Behavior at PP=1 unchanged (24 of 24 policy scenarios). Against a real vLLM 0.10.2 DP2 PP2 deployment (G4), Frontier's balancer reproduces all 48 formal routes from the native history (C3). C3 feeds the native reports into the balancer, so it checks the balancer, not Frontier's own report emission; that comparison is a step of the S43 task. The discriminating slice was not reached (C4 `SCENARIO_NOT_REACHED`, plan §18.21). The G3b review fixes (`dfb0b25`, `20f0f94`, `df8ebc6`) publish lane load after stale drops and after deferred terminal releases, and remove guards for unreachable states; `test_report_2026-09-24_g3b_fixes.md`. |
| W9-04 | A lane that joins a forward after receiving a first-layer placeholder no longer stalls it | Landed. `2ffb062`; checks A1–A7 pass, including fidelity 71 of 71 and stage-admission 51 of 51. |
| Fix review (2026-09-24) | Review of W2, W3, W4, W6, W7, W9 and W9-05 as landed | Ten commits `c647e95`..`6aee289`: PP>1 decode preemption (F-R1..F-R4), random-policy lane collapse (W2-R1), unreachable guards and a single phase rule (W3), FP8 wiring (W6-R1..R4), cross-server empty all-to-all (W7-R1, companion `ff11ee6`), the reference loop's idle iteration and two stagger cases (W9). Proposals F-R5, F-R6, W3-R5, W3-R6, S44 await decisions. `test_report_2026-09-24_fix_review.md`. |
| W9-05 | A `vllm_v1` MONOLITHIC request preempted during decode resumes instead of disappearing | Landed. `75c1140`; checks B1–B8 pass. A 72-cell KV-pressure sweep lost 176 requests before and none after; fidelity 71 of 71 identical. |

## Deliverables

### Source

| Path | Change |
| --- | --- |
| `frontier/scheduler/cluster_scheduler/round_robin_cluster_scheduler.py` | W2: DP lane index derives from the persistent counter, so rotation survives across calls |
| `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py` | W4: shared lane-resolution seam |
| `frontier/scheduler/cluster_scheduler/vllm_load_balancing_cluster_scheduler.py` | W4: the new opt-in policy (new file) |
| `frontier/scheduler/utils/vllm_dp_load_balancer.py` | W4: the delayed-snapshot load model (new file) |
| `frontier/scheduler/cluster_scheduler/cluster_scheduler_registry.py`, `frontier/types/cluster_scheduler_type.py`, `frontier/config/cluster_scheduler_config.py`, `frontier/config/config.py` | W4: registration through the existing registry |
| `frontier/scheduler/utils/forward_collective.py` | W3: the shared monolithic forward (new file) |
| `frontier/scheduler/utils/sync_entry.py`, `sync_state.py`, `forward_sync_state.py`, `prefill_collective.py`, `decode_collective.py`, `ep_wave_schedule.py`, `ep_wave_inputs.py` | W3: one lifecycle for mixed-source cohorts |
| `frontier/events/cluster_schedule_event.py`, `frontier/events/global_batch_end_event.py`, `frontier/scheduler/request_load.py`, `frontier/scheduler/replica_scheduler/*` | W3/W4: event and load-snapshot wiring |
| `frontier/profiling/moe/moe_vllm_kernel.py` | W6: the legacy path performs the gated expert computation |
| `frontier/cc_backend/backends/collective-sim` | W7: gitlink moved from `b8518af` to `eb7bc4f`, then to `ff11ee6` (`6d621c8`, empty cross-server all-to-all) and `e922c77` (`9adf759`, zero refused for every other kind); after companion PR 1 merged, `d28fe917` on companion `main` (`1b11eff`, same tree) |
| `docs/profiling/README.md` | W6: the operator's scope and its artifact-identity limits |
| `frontier/scheduler/cluster_scheduler/vllm_load_balancing_cluster_scheduler.py`, `base_cluster_scheduler.py`, `frontier/scheduler/replica_scheduler/base_replica_scheduler.py`, `frontier/scheduler/replica_stage_scheduler/stage_execution_context.py` | W9: schedule-time load reports while the pipeline has room, keyed by the stage-0 forward group |
| `frontier/scheduler/utils/sync_entry.py` | W9-04: withdraw a first-layer placeholder when its lane joins the forward |
| `frontier/scheduler/replica_scheduler/vllm_v1_kv_allocation.py` | W9-05: preemption resets token progress only for a victim still in prefill |

### Tests

| Path | Purpose |
| --- | --- |
| `tests/unit/test_cluster_scheduler_dp_lanes.py` | W2 placement, extended to state where each request lands |
| `tests/unit/test_monolithic_mixed_forward_sync.py`, `tests/integration/test_monolithic_mixed_forward_runtime.py` | W3, unit and real event loop |
| `tests/unit/test_vllm_dp_load_balancer.py`, `tests/integration/test_vllm_dp_placement_runtime.py` | W4 and W9, unit and real event loop; the runtime module also carries the W9-04 regression case |
| `tests/unit/test_dp_placement_reference_loop.py` | W9: the vLLM engine-iteration reference loop |
| `tests/integration/test_vllm_v1_decode_preemption_runtime.py`, `tests/unit/test_pdaf_decode_attn_preemption.py` | W9-05: a request preempted during decode resumes and completes; a victim still in prefill restarts |
| `tests/unit/test_moe_fused_expert_arithmetic.py` | W6 on CPU, against plain-Torch references |
| `tests/integration/test_moe_fused_expert_numerical_parity.py` | W6 against vLLM's own `fused_experts` on a GPU |
| `tests/unit/test_collective_sim_zero_payload.py` | W7 through the Frontier backend boundary |
| `tests/frontier_sources.py` | Enumerates Frontier-owned sources so governance scans skip the vendored submodule |

Companion repository: `tests/test_zero_payload_input.py`, 9 tests, published with
a narrowed `.gitignore` so it ships while private working material does not.

### Records

All under `task_memory/task_2026-09-21_issue26_correctness_pr/`: `plan.md`,
`requirements.md`, `progress.md`, `review.md`, `validation.md`, `design.md`,
`future.md`, `summary.md`, three pinned-source audit reports, five test reports
(W3, W4, W6, W7, and this Step 8 combined regression; W2's measurement is
recorded in `validation.md`), and `w5_reverted_moe_routing_runtime_path.patch`.

## Observed validation results

| Check | Result |
| --- | --- |
| Unit suite at `d881357` | 84 failed, 3782 passed, 49 skipped, 11 errors. The `FAILED` set is identical to the recorded `origin/main` baseline in both directions. |
| Integration suite | 15 passed, 22 skipped, 5 errors. The added skip is the W6 GPU module; the 5 errors are the absent pinned PD-AF Reference checkout and are identical on the base. |
| Architecture examples | 16 of 16 pass, covering co-location, sequential PDD, and sequential PD-AF in offline and online modes. |
| Pipeline cases | 4 of 4 `PP=2` runs pass. |
| Predictor cache, cold then warm | Cold 26.9 s writing 63 artifacts; warm 2.1 s writing none; `request_metrics.csv` byte-identical. |
| W3 fidelity matrix | 71 of 71 cases identical, against an expectation recorded before the run. |
| W4 fidelity matrix | 71 of 71 cases identical, against an expectation recorded before the run. |
| W4 placement | Measured to place differently from round-robin under the same load, so the policy is not a renamed default. |
| W6 native parity | Eight native tests passed on an H800 (`exp-0922-145047-660565`, charged group `codesign`, vLLM 0.10.2, `VLLM_API_VERSION=0.10.x`): seven reference-output comparisons at `rtol=0, atol=0` and one FP8 structural/finite-output check. FP8 numerical equivalence is not established. The FP8 check as run omitted the production `block_shape`; corrected 2026-09-22 (C35-03) and rerun natively (`exp-0922-202645-561899` at `c231322`, 8 passed, `block_shape=[128, 128]`). `f236c17` later changed the FP8 step (fix review W6-R1..R3); the native rerun `exp-0924-114241-429126` at `363a1dd` covers it (8 passed, FP8 structural only). CPU tests 32 passed after `6828581` pins the profiler's tile-config and alignment arguments (W6-R6). |
| W7 companion | 9 passed on the fix; 6 of 9 fail against pristine sources. |
| W7 Frontier | 4 passed; 3 of 4 fail at the old gitlink. A fresh clone resolves `eb7bc4f` from the published remote, builds, and passes. Superseded 2026-09-24: companion `ff11ee6` defines the cross-server empty all-to-all (W7-R1); Frontier's three rewritten cases pass and fail at `eb7bc4f`; companion 12 passed. |
| Step 9 ground truth | G3 (`exp-0923-221233-009652`, PP1) T1 38/38; G4 (`exp-0923-230103-591735`, DP2 PP2 EP, 4 H800) extraction PASS; G5 T1 48/48 formal routes MATCH; C4 `SCENARIO_NOT_REACHED` in all four bursts; the pre-change revision rejects PP2 at construction (`test_report_2026-09-23_step9_dp_pp_groundtruth.md`). |
| Negative controls | W2 12 of 23, W3 four trees, W4 five trees, W6 one discriminating test, W7 both sides — each fails for its own stated reason on the unrepaired source, except W6: there its tests error at fixture setup, and a mutant restoring only the slice arithmetic fails the discriminating test on its assertion (corrected 2026-09-24). |

Detailed commands, expectations, and limits are in `validation.md` and the
per-work-package reports.

## Open and deferred work

| Item | Where |
| --- | --- |
| The pre-existing `tests/debug/` pointer defect: `AGENTS.md` §Tests, a docstring at `vllm_v1_engine_replica_scheduler.py:16`, and 10 of the 84 baseline unit failures all reference a tree that exists neither here nor on `main`. Reported, not repaired; its fix is a decision about the published test surface. | `future.md` §1 |
| Retarget PR 35's base to `main` once PR 34 merges. | PR 35 description |
| Issue 26 itself stays open; this PR is a subset of it. | PR 35 description |
| Step 9 C4: decided 2026-09-24, option (a), `SCENARIO_NOT_REACHED` accepted. S43 (PP>1 admission after an empty schedule) and S42 (DP dummy forwards) continue as separate calibration-and-repair tasks. | `task_memory/task_2026-09-24_s43_pp_empty_schedule_admission/`, `task_memory/task_2026-09-24_s42_dp_wave_idle_forward/` |
| Fix-review proposals: F-R5 victim selection (stacked draft PR 38, `fix/vllm-v1-preemption-victim`), F-R6 in-flight token and resumed-victim replay (CPU measurement), S44 (row of the S43 task), the pinned calibration tools. W3-R5 and W3-R6 are fixed in `341970d` (merge `1978b72`). The native W6 FP8 rerun ran on 2026-09-24 (`exp-0924-114241-429126`, 8 passed). W7-R4 was decided (option a) and fixed in `9adf759`. | `test_report_2026-09-24_fix_review.md` §8 |
| W9-05 follow-ups, not started: at PP>1 the fix review's F-R1..F-R4 are fixed (`c647e95`); the recompute cost of a resumed request is not modeled; the waiting loop still drops a request with `num_new_tokens <= 0` silently where vLLM asserts; MONOLITHIC preemptions appear only in `request_total_preemption_count`. | `issues.md` W9-05, Limits |

## Limits of what was validated

CPU only, apart from the three W6 native parity jobs (`exp-0922-145047-660565`, `exp-0922-202645-561899`, `exp-0924-114241-429126`) and the Step 9 ground-truth jobs G3 and G4. No native profiling suite was
run as a gate, and no vLLM serving or TTFT comparison was performed — both are
outside this task and neither is needed to accept the PR. The PD-AF
Reference-checkout integration tests could not run on this host. Apart from the
two CSV smokes, the example runs use dummy execution time, so they validate
structure, lifecycle, and conservation rather than latency accuracy. The final
diff review was a self-review by the same agent that wrote the change, not an
independent one; `review.md` states this.
