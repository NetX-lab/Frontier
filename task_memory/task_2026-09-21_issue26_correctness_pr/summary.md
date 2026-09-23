# Issue 26 Correctness PR — Summary

## Modification History

| Date | Change |
| --- | --- |
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
| [fwyc0573/frontier-htsim#1](https://github.com/fwyc0573/frontier-htsim/pull/1) | `fix/zero-payload-input-handling` | `main` @ `b8518af` | Companion backend fix, commit `eb7bc4f` |

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
| W9 | The opt-in vLLM DP placement policy supports pipeline parallelism (Step 9) | Landed on the CPU: `2ffe78d`, tests `bacdbb4`. Behavior at PP=1 unchanged (24 of 24 policy scenarios). The vLLM-side GPU comparison G3–G5 is not run. |
| W9-04 | A lane that joins a forward after receiving a first-layer placeholder no longer stalls it | Landed. `2ffb062`; checks A1–A7 pass, including fidelity 71 of 71 and stage-admission 51 of 51. |
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
| `frontier/cc_backend/backends/collective-sim` | W7: gitlink moved from `b8518af` to `eb7bc4f` |
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
| W6 native parity | Eight native tests passed on an H800 (`exp-0922-145047-660565`, charged group `codesign`, vLLM 0.10.2, `VLLM_API_VERSION=0.10.x`): seven reference-output comparisons at `rtol=0, atol=0` and one FP8 structural/finite-output check. FP8 numerical equivalence is not established. The FP8 check as run omitted the production `block_shape`; corrected 2026-09-22 (C35-03), native rerun NOT_RUN. |
| W7 companion | 9 passed on the fix; 6 of 9 fail against pristine sources. |
| W7 Frontier | 4 passed; 3 of 4 fail at the old gitlink. A fresh clone resolves `eb7bc4f` from the published remote, builds, and passes. |
| Negative controls | W2 12 of 23, W3 four trees, W4 five trees, W6 one discriminating test, W7 both sides — each fails for its own stated reason on the unrepaired source. |

Detailed commands, expectations, and limits are in `validation.md` and the
per-work-package reports.

## Open and deferred work

| Item | Where |
| --- | --- |
| Re-point the collective-sim gitlink at `main` once companion PR 1 merges. `.gitmodules` already records `branch = main`; `git submodule update --remote` would currently drop the fix. | `future.md` §2 |
| The pre-existing `tests/debug/` pointer defect: `AGENTS.md` §Tests, a docstring at `vllm_v1_engine_replica_scheduler.py:16`, and 10 of the 84 baseline unit failures all reference a tree that exists neither here nor on `main`. Reported, not repaired; its fix is a decision about the published test surface. | `future.md` §1 |
| Retarget PR 35's base to `main` once PR 34 merges. | PR 35 description |
| Issue 26 itself stays open; this PR is a subset of it. | PR 35 description |
| Step 9 vLLM-side comparison G3–G5: authorized by the user on 2026-09-23 (≤3 jobs × ≤1 h, `codesign`); G2 harness in preparation. | plan §18.5, §18.6 |
| W9-05 follow-ups, not started: the recompute cost of a resumed request is not modeled; the waiting loop still drops a request with `num_new_tokens <= 0` silently where vLLM asserts; MONOLITHIC preemptions appear only in `request_total_preemption_count`. | `issues.md` W9-05, Limits |

## Limits of what was validated

CPU only, apart from the one W6 GPU parity job. No native profiling suite was
run as a gate, and no vLLM serving or TTFT comparison was performed — both are
outside this task and neither is needed to accept the PR. The PD-AF
Reference-checkout integration tests could not run on this host. Apart from the
two CSV smokes, the example runs use dummy execution time, so they validate
structure, lifecycle, and conservation rather than latency accuracy. The final
diff review was a self-review by the same agent that wrote the change, not an
independent one; `review.md` states this.
