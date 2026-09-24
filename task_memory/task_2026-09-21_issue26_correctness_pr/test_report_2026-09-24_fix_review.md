# Test report — review of the branch's existing fixes (2026-09-24)

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-24 | Section 8 and 9: the native W6 FP8 rerun ran (`exp-0924-114241-429126`, 8 passed). |
| 2026-09-24 | W7-R4 fixed by the user's choice of option (a): companion `e922c77`, gitlink `9adf759`; sections 3, 4 and 8 updated. |
| 2026-09-24 | Created: review findings and dispositions, verification of each fix, the dummy-mode answer, candidate row S44, final validation. |
| 2026-09-24 | Workflow `wf_7606e14e-f10` results reconciled (section 10): W6-R6 fixed (`6828581`); W6-R7, W7-R4 and W9-R4 added; W7-R4 proposal in section 8; a blind-spot row in section 5; section 9 corrected. |
| 2026-09-24 | Section 7 filled: all five pre-recorded expectations met. W3-R3 wording corrected (the removed scan guarded an unreachable state; no group-formation check exists) and the matching `3a8767c` erratum added. |

## 1. Scope and method

Request (2026-09-24, verbatim in `requirements.md`): "对已有修复进行一次review并修正review中发现的问题", together with the question whether correcting and repairing logic in dummy mode alone is sound.

Reviewed packages: W2 (round-robin DP rotation), W3 (shared monolithic forward), W4 (opt-in vLLM DP placement), W6 (legacy fused-MoE profiling), W7 (zero-payload collective-sim), W9 (PP>1 DP placement) with W9-04 and W9-05, and cross-package effects.

Reviewers:
- A multi-agent workflow ran first. Most of its agents stopped at a platform rate limit after exploring, so their notes are leads, not findings.
- The main session then reviewed every package itself and treated each lead as unverified.

Rules:
- No finding was fixed until source reading or execution confirmed it.
- Every fix that carries a test was checked against a tree without the fix (a negative control).
- Findings that change simulated behavior beyond a correctness defect, or that exceed the approval limits, are proposals (section 8), not commits.

Baseline: `ba0a804`, the pushed PR 35 head before the review. Review commits: `c647e95` through `6aee289`.

## 2. Environment

| Item | Value |
| --- | --- |
| Host | `kun-workspace-vgen2`, CPU only, 64 cores |
| Simulator Python | `/data/ycfeng/envs/frontier-py310/bin/python` (Python 3.10) with `PYTHONPATH=<tree> WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp` |
| Torch tests | `/data/ycfeng/envs/openmopd-py312/bin/python` (Torch 2.8.0, vLLM 0.11.0; the vLLM kernels are replaced by CPU references inside the tests) |
| vLLM reference | `/data/ycfeng/Frontier/.real-engine/vLLM-BS` (v0.10.2 source; checkout `63ac6c6b9`, instrumentation on top of `ea95f571e`) |
| Evidence root | `/data/ycfeng/tmp/issue26-correctness-pr/review_20260924/` and `/data/ycfeng/tmp/issue26-correctness-pr/final_20260924/` |

## 3. Findings and dispositions

| Id | Package | Finding | Confirmed by | Disposition |
| --- | --- | --- | --- | --- |
| F-R1 | W9-05 | At PP>1 a decode request preempted while an earlier batch still carried it kept its active-batch mark. The running phase then skipped it for good and the run stalled. | `w9_05_regress/negctl`: `dense_pp4` ends with a non-empty scheduler state | Fixed, `c647e95` |
| F-R2 | W9-05 | The stale in-flight step still credited its layers, so the resumed step overran the layer counter. | `moe_dp2_ep2_pp2`: `ValueError: Decode post_moe layer counter cannot advance` | Fixed, `c647e95` |
| F-R3 | W9-05 | The old batch's end released the victim a second time after its new batch admitted it, so the victim was scheduled while in flight. | Rescheduling assertion in the integration test | Fixed, `c647e95` |
| F-R4 | W9-05 | At PP>=4 a finished victim held for its terminal release re-entered the waiting queue. | `dense_pp4_finished_victim` | Fixed, `c647e95` |
| F-R5 | W9-05 | Victim selection excludes the requester. vLLM takes `running[-1]` (or the lowest priority), which can be the requester, and then stops scheduling it (`scheduler.py:470-548`). Frontier passes `exclude=request` (`vllm_v1_kv_allocation.py:412-465`, `:661`). | Source reading | Fidelity difference; proposal |
| F-R6 | W9-05 | The token a preempted request's in-flight step samples is discarded. vLLM's `update_from_output` still applies it, and a request that stops there is retired from waiting (`scheduler.py:1278-1324`). | Source reading | Fidelity difference; proposal |
| W2-R1 | W2 | `RandomClusterScheduler` took the DP lane from the index inside the current call. Online arrivals come one per call, so every request landed on lane 0. | Placement test fails before the fix | Fixed, `3ec7bbf` |
| W2-R2 | W2 | Batch mode copied the decode role's rotation, so one rule was written twice. | Code reading | Single rotation, `3ec7bbf`; results unchanged by the matrix |
| W2-R3 | W2 | Two guard tests could not fail for the defects they named: a source-text check the pre-fix line satisfied, and set arithmetic that never called `schedule()`. | Code reading | Removed, `3ec7bbf` |
| W2-R4 | W2 | AGENTS.md described cyclic placement for every role; PD-AF `DECODE_ATTN` places by load. | Code reading | AGENTS.md scoped, `3ec7bbf` |
| W2-R5 | W2 | The records claim no wrapper can give a MoE role more than one lane. The wrappers' pass-through flags can, because they are appended after the wrapper's `ATTN_TP == MOE_TP * MOE_EP` check. | Matrix rows run | Matrix rows `748e757`; records corrected in `design.md`, `validation.md` and `review.md` |
| W3-R1 | W3 | Per-source continuation changes PDD PREFILL and DECODE timing when `attn_dp>1`, and no test pinned it. | Mutation control `w3_borrowed`: 2 failed, 23 passed | Unit test `3a8767c` and trained row `748e757` |
| W3-R2 | W3 | The legacy-marker loop in `handle_forward_sync_collective` guarded a state every source now excludes. Its comment named a prefill-helper check that does not exist. | Code reading | Removed, `3a8767c` |
| W3-R3 | W3 | The per-layer duplicate-owner scan in `prepare_ep_wave_inputs` ran at every operator for a state no path produces: `ClusterScheduleEvent` adds each request to exactly one lane's replica scheduler (`cluster_schedule_event.py:56-64`), and each lane builds batches only from its own requests. Only a hand-built test with one request in two lanes' batches reached it. | Code reading | Removed, `3a8767c`; reverses the `audit_scheduler.md` PORT row |
| W3-R4 | W3 | `source_forward_mode` claimed to be the stage-schedule event's rule while the event classified phases inline. | Code reading | Single source, `3a8767c` |
| W3-R5 | W3 | `sync_entry` admits by a looser predicate than the one group formation uses. | A strict predicate caused 32 regressions in 9 test files (`strict_pred_compare.json`) | Proposal |
| W3-R6 | W3 | The alias rooms and per-kind partition in `sync_state.py:39` duplicate state. | Code reading | Refactor across more than five files; proposal |
| W4-R1 | W4 | A lane publishes requests routed to it during its forward as waiting. vLLM drains its input queue only at the top of each busy-loop iteration, so those requests are absent from the counts published at the end of that iteration (section 6). | Source reading plus native G4 receipts | New candidate row S44; proposal |
| W6-R1 | W6 | The FP8 kernel config was looked up for the unquantized dtype. | Negative control: `[(bf16, {})] != [(bf16, {'use_fp8_w8a8': True})]` | Fixed, `f236c17` |
| W6-R2 | W6 | FP8 accumulated in FP16 whatever the hidden-state dtype. | Negative control: `tl.float16 == tl.bfloat16` fails | Fixed, `f236c17` |
| W6-R3 | W6 | The hidden state was quantized before the timed step, and a `.contiguous()` copy could receive the kernel's writes. | Code reading against `fused_experts_impl` | Fixed, `f236c17` |
| W6-R4 | W6 | The import comment promised a fallback to the functional path that cannot happen. | Code reading | Fixed, `f236c17` |
| W6-R5 | W6 | `docs/profiling/README.md` dated the completeness fix by calendar and claimed every later path includes the reduction; `frontier_loop` does not. | Code reading | Fixed, `b7a7ac5` |
| W6-R6 | W6 | The native parity test copies the profiler's tile-config and alignment calls instead of calling them, and the CPU stubs ignored their arguments, so nothing checked the block shape, token count, global expert count or expert map `profile_fused_moe_kernel` passes. | Mutants m5 (`block_shape=None`), m6 (local expert count), m7 (`expert_map=None`): 31 earlier tests pass on each | Test added, `6828581` |
| W6-R7 | W6 | Records: `validation.md` said all 8 native cases compare at zero tolerance (seven do; FP8 is structural); `review.md` said the native test uses the profiler's setup (it copies it); `summary.md` and `review.md` said the W6 negative control fails for its stated reason on the unrepaired source (there the tests error at fixture setup). | `7269bac^` run: 1 failed, 6 errors; slice-restoring mutant: the discriminating test fails on its assertion | Records corrected |
| W7-R1 | W7 | `eb7bc4f` let a zero payload through validation, but the runner's all-to-all generator was undefined at zero once a pair crossed servers: `ZeroDivisionError` (`pairwise_steps`), a hang (`nccl_pairwise`, htsim reads a 0-byte flow as unbounded, `ndp.cpp:934`), or 0 ms (`full_mesh`, one phase). | Probes in `w7_fix/before` and `negctl` | Fixed in the companion (`ff11ee6`) and the gitlink (`6d621c8`) |
| W7-R2 | W7 | The Frontier test covered only one server. A reduce-scatter test covered a method no Frontier path calls, and a negative-payload test covered a guard shared by every backend. | Code reading | Test rewritten, `6d621c8` |
| W7-R3 | W7 | After `beded3c` the config-name scan still skipped sources that fail to parse, which would now hide an owned module. | Code reading | Fixed, `7309f5d` |
| W7-R4 | W7 | A zero-payload cross-server all-reduce hangs in the companion runner: the ring's per-edge sizes reach 0 bytes, which htsim treats as unbounded. `eb7bc4f` made zero legal for every collective kind; `ff11ee6` defines zero only for all-to-all. No Frontier path sends it (payloads come from batches with at least one token, idle placeholders are not priced, a zero MoE-TP all-reduce is `EXACT_NOOP`). | Probe, 2 x 8 GPUs, DP=16 ring: 480 zero-size connections on edges `7->8` and `15->0`; stopped advancing at simulated time 10.88, payload 1 passed 271 | Fixed by user decision, option (a): companion `e922c77`, gitlink `9adf759` |
| W9-R1 | W9 | The reference oracle raised when an engine had no work, and a test pinned per-engine step drift. vLLM runs `execute_dummy_batch()` for an idle engine in a running wave, and `_has_global_unfinished_reqs` advances its counter (`core.py:1170-1216`). | Source reading | Fixed, `8ca0387`; plan row superseded |
| W9-R2 | W9 | Every PP>1 case arrived as a burst, so no lane published an admission into a forward its peer had already started. A key that always named the next forward passed them all. | Mutation M2 (section 4) | Two stagger cases, `6aee289` |
| W9-R3 | W9 | `Publication` and `ReferenceDeployment.publications` were unused. | Code reading | Removed, `8ca0387` |
| W9-R4 | W9 | S9-08 overstated the integration test's independence: its classifier `running_after >= num_pipeline_stages` restates the policy's room rule, and the reference oracle is not compared with Frontier's reports. A change applied to both the policy and the classifier passes every W9 test. | Code reading | Record qualified (`review.md` S9-08); the oracle comparison becomes a step of the S43 task |

Commit-message errata (history is not rewritten):
- `f236c17` says the local-reduction unit test "now derives its expected value from the inputs". It was deleted, because it asserted the CPU stub's own definition.
- `3a8767c` says group formation "already refuses" a request owned by two lanes. No check does; the scan was removed because the state is unreachable (W3-R3), which is also the reason the removed test had to build that mapping by hand.
- `6aee289` says "the burst cases pass" under the mutation. Precisely: the PP>1 burst cases pass, and the PP=1 asymmetric-trace case `test_placement_follows_published_load_where_round_robin_cannot` fails.

## 4. Verification of each fix

| Fix | Command / evidence | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| `c647e95` | `pytest tests/integration/test_vllm_v1_decode_preemption_runtime.py` on HEAD and on control trees (`w9_05_regress/negctl/matrix.txt`) | Each case fails on the tree that lacks the fix named in its comment | Control (no fix): 3 failed, 1 passed. Without the finished-victim fix: `dense_pp4_finished_victim` fails. Without the layer reset: `moe_dp2_ep2_pp2` fails. HEAD: 4 passed | PASS |
| `c647e95` | `w9_05_regress/probe.py`, 72 KV-pressure cells, PP 2 and 4, dense and MoE DP2 EP2 | Every cell drains with all tokens and no double scheduling | 72 of 72 drained, 0 short, 0 double-scheduled (section 7, `final_20260924/probe_head.log`) | PASS |
| `3ec7bbf` | `pytest tests/unit/test_cluster_scheduler_dp_lanes.py tests/unit/test_replica_identity_contract.py tests/unit/test_vllm_dp_load_balancer.py`; before/after E2E in `w2_random/` | The online random policy uses both lanes; batch-mode placements unchanged | Tests pass. Before the fix every online request is on lane 0 | PASS |
| `3a8767c` | `pytest tests/unit/test_monolithic_mixed_forward_sync.py` on HEAD and on a tree whose continuation borrows the first lane's duration (`w3_borrowed/result.txt`) | The new parametrized test fails on the borrowed tree for both roles | 2 failed (prefill, decode), 23 passed on the borrowed tree; all pass on HEAD | PASS |
| `748e757` | `run_matrix.py run --case-filter dp_moe_` on `2310417`, `3d47417`, a borrowed tree and HEAD (`newcases/`) | The co-location row fails before the shared forward; the PDD rows move with W2; the trained row separates borrowed from per-lane timing | `3d47417` exits 1 with "non-empty scheduler state" (also at QPS 0.5 and 2.0). Trained PDD mean TPOT: borrowed 5.21995, HEAD 5.23333 ms (+0.26 %); E2E 58.2919 vs 58.4640 ms (+0.30 %). The dummy PDD row is identical between borrowed and HEAD | PASS |
| `f236c17` | `pytest tests/unit/test_moe_fused_expert_arithmetic.py tests/unit/test_moe_fused_event_contract.py` (openmopd) on HEAD and on `f236c17~1` with the new tests (`final_20260924/w6_negctl.txt`) | Behavior tests fail on the pre-fix profiler | HEAD 31 passed. Pre-fix: 10 failed, 21 passed. The FP8 config lookup (2) and the compute type (1) fail on behavior; 7 fail on the removed `block_dims` argument, an interface change rather than a behavior check | PASS, with the interface caveat |
| `6d621c8` | Companion `tests/test_zero_payload_input.py`; Frontier `tests/unit/test_collective_sim_zero_payload.py`; controls at `eb7bc4f` | Cross-server empty all-to-all priced like one byte and above zero | Companion 12 passed; Frontier 3 passed (`final_20260924/collective_sim_*.txt`). At `eb7bc4f`: companion `pairwise_steps` RuntimeError, `full_mesh` AssertionError, `nccl_pairwise` timeout; Frontier 2 failed (`ZeroDivisionError`), 1 passed | PASS |
| `7309f5d` | `pytest tests/unit/test_module_split_boundaries.py` | Passes; no parse error is skipped | Passes in the HEAD unit suite; no regression by id (section 7) | PASS |
| `8ca0387` | `pytest tests/unit/test_dp_placement_reference_loop.py tests/unit/test_vllm_dp_load_balancer.py` | Idle iteration recorded; peers aligned | 88 passed | PASS |
| `6828581` | `pytest tests/unit/test_moe_fused_expert_arithmetic.py tests/unit/test_moe_fused_event_contract.py` (openmopd) on HEAD and on three single-argument mutants of the profiler (`wf_7606e14e_f10/``w6_negative_controls.txt`) | Each mutant fails only the new test | HEAD 32 passed; m5, m6, m7 each 1 failed, 31 passed, the failure being `test_the_tile_config_and_the_alignment_follow_fused_experts` | PASS |
| `9adf759` (companion `e922c77`) | Companion `tests/test_zero_payload_input.py` on `e922c77` and on `ff11ee6`; the cross-server probe on `e922c77`; Frontier `tests/unit/test_collective_sim_zero_payload.py` (`wf_7606e14e_f10/w7_r4_zero_payload_by_kind.txt`) | The four new cases fail on `ff11ee6`; zero is refused for allreduce, allgather, reducescatter and p2p at once; all-to-all unchanged | `e922c77` 16 passed; `ff11ee6` 4 failed, 12 passed. Probe: RuntimeError with runner exit 2 for all three ring kinds, where the old all-reduce ran about 9 minutes. Frontier 3 passed | PASS |
| `6aee289` | `pytest tests/integration/test_vllm_dp_placement_runtime.py` on HEAD, and on an export with `joinable_forward_group_id` forced to `_next_forward_group_id` (M2, `review_20260924/w9_stagger/`) | HEAD passes. Under M2 both stagger cases fail and the PP>1 burst cases pass | HEAD 12 passed. M2: 3 failed (both stagger cases and the PP=1 asymmetric case), 88 passed with the balancer suite. The premise helper counts one reported join in each stagger case and zero in every burst case | PASS |

## 5. Is logic correction and repair in dummy mode alone sound?

**Answer: for control-flow rules only, and not for closing a calibration repair.**

Dummy mode prices every operator at `dummy_execution_time_ms` whatever its shape. That keeps the event order and state transitions, but it removes three things: duration differences between lanes, the effect of load imbalance on time, and every branch that timing selects.

What dummy mode did find in this branch:
- W9-05 F-R1 to F-R4. These fire whenever a preemption lands while an earlier batch is in flight, at any timing. The pre-fix error appeared under every timing tried.
- The random-policy lane collapse (W2-R1).

What it cannot see (measured here):

| Blind spot | Evidence |
| --- | --- |
| Per-lane durations | The dummy PDD two-lane row is identical between a tree that borrows the first lane's duration and HEAD. The trained row differs (TPOT +0.26 %) (`newcases/`). |
| Load imbalance | Before `3ec7bbf` the online random policy put every request on lane 0, yet dummy mean TTFT was only 1.1 % higher than after the fix (1995.73 vs 1973.56 ms, `w2_random/`). |
| Timing-selected branches | G4 case, burst a. In dummy mode both lanes' first admissions report under key 12 in one stage-0 group, because the first MoE sync is at 6.51 ms. In non-dummy mode lane 0's layer-0 EP wave starts alone at 0.242 ms, before lane 1 arrives at 0.25 ms, so the keys are 12 and 13 (`calib_nondummy/analysis/`). |
| The steady segment | The case records classify differences in the bursts only. In the steady segment lane agreement is 13 of 24 under dummy timing and 19 of 24 non-dummy; snapshot-count agreement is 6 and 7 of 24 (workflow `wf_7606e14e-f10`, `workflow_gap_summary.md` errata). |
| Fidelity of a rule | F-R5 and F-R6 are differences in what is computed, not control-flow defects; only a timed comparison can size them. |

Why dummy mode cannot close a calibration repair:
- Diagnostic E2E relative errors on the 48 formal G4 requests (`calib_nondummy/analysis/e2e_diag.json`):
  - dummy: TTFT 0.603, TPOT 3.34, E2E 3.17;
  - non-dummy fallback: 0.716, 0.78, 0.78.
  - The contract gate is at most 0.10 per metric.
- The pinned tools the contract requires, the E2E normalizer and the op-supplement merge tool, are absent. `/data/ycfeng/frontier-calibration-old-20260831/` does not exist, and the contract allows no substitute.
- S39 routing is `UNSET`.

The non-dummy fallback is not usable as it stands:
- The checked-in `h800/Qwen3-30B-A3B-tiny` profiles cover at most 128 tokens for linear ops and 64 for MoE. The G4 chunk is 20448 tokens, so the random forest extrapolates.
- `balanced` routing fails: the CSV holds only `routing_runtime_path=uniform_topk` rows.
- The resulting non-dummy timing underestimates the long chunk by about 55 % and small forwards by 4 to 6 times.

What this means for the work:
- Dummy runs, reference loops and unit fixtures are adequate for the control-flow repairs made here. Each such fix was checked against the vLLM source, not against a dummy-timed number.
- A per-lane timing change, such as W3's PDD continuation, needs a trained or native-timed row. The trained row was added.
- S43 and S42 change when work is admitted or executed. Their effect is a time, so they need shape-dependent timing (native iteration durations or profiles that cover the case) and the pinned E2E tools before they can close. Under dummy timing the S43 delay would be about 111 ms per small forward, against 21–47 ms native, which amplifies it 3 to 5 times.

## 6. Candidate row S44 — published counts include undrained input-queue requests

Reference (`vllm/v1/engine/core.py`, v0.10.2):
- `DPEngineCoreProc.run_busy_loop` (`:1170-1216`) calls `_process_input_queue()` at the top of each iteration.
- It then steps the engine, and only then calls `_maybe_publish_request_counts()`, which publishes `scheduler.get_request_counts()` (`:1130-1144`).
- A request that reaches the engine while it is inside the step waits in `input_queue`, so the counts published at the end of that iteration do not include it.
- The frontend's local reservation for that request is replaced when those counts arrive.

Frontier: `VllmLoadBalancingClusterScheduler.on_replica_batch_end` publishes `lane.get_request_load()`. That load includes every request routed to the lane since it last scheduled.

Measured:

| Source | Reports | With `waiting > 0` |
| --- | --- | --- |
| Native G4 coordinator receipts | 92 | 1 |
| Frontier G5 policy run (`w4_input_queue/runs/baseline_pp2`) | 106 changed-count reports | 24 |
| Counterfactual that subtracts requests routed since the lane's last admission (`w4_input_queue_cf/counterfactual.diff`) | 96 | 0 |

- The counterfactual does not improve placement agreement on G4: steady-segment mismatches go from 11 to 13 of 24. The dummy timing differences (WG11) dominate.
- A PP=1 grid (`w4/midstep/`, 240 cells) places differently in 12 cells when the undrained requests are hidden.

Status: a confirmed semantic difference with no measured placement benefit yet. The counterfactual models only iterations that schedule a batch. An iteration that drains the queue but schedules nothing would need its own start point. The mechanism is the same input queue that S43 concerns, so this row is proposed for the S43 task (section 8) rather than fixed here.

## 7. Final validation

Baseline `ba0a804` against HEAD `6aee289`, each on a clean detached worktree
(`.worktrees/review-final-base`, `.worktrees/review-final-head`). Script:
`final_20260924/run_final.sh`, 2026-09-23T18:17:52Z to 18:33:08Z. Expectations
were written first, in `final_20260924/expectations.md`.

| # | Check | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| 1a | Unit suite, `pytest tests/unit --continue-on-collection-errors` with JUnit, `frontier-py310`; `composition_compare_junit.py` | 0 regressions, 0 new failures; id changes are only the review's tests | 84F/3829P/51S/10E → 84F/3828P/51S/10E. 0 regressions, 0 new failures, 0 skip changes. Only before (6): W2-R3's removed guard, W9-R1's two replaced reference-loop tests, W3-R3's duplicate-owner test, W3-R2's two marker cases. Only after (5): W2-R1's random-policy test, W9-R1's two reference-loop tests, W3-R1's two per-lane continuation cases | PASS |
| 1b | Integration suite, same method | Same | 28P/22S/5E → 33P/22S/5E. 0 regressions. Only after: the two stagger cases (W9-R2) and three preemption cases (F-R1..F-R4). The 5 errors are the absent pinned PD-AF Reference checkout on both sides | PASS |
| 1c | Modules the suites cannot run here | Pass at HEAD | `tests/unit/test_collective_sim_zero_payload.py` skips at collection in the clean worktrees (submodule not initialized). Run in the branch worktree with the submodule at `ff11ee6` and built: Frontier 3 passed; companion `tests/test_zero_payload_input.py` 12 passed. The W6 files need Torch: `openmopd-py312` 31 passed. Rerun 2026-09-24 with the commands below, same results (`final_20260924/rerun_separate/`) | PASS |
| 2 | Fidelity matrix, `run_matrix.py run` on both trees, then `compare` | 74 of 74 identical | 74 of 74 identical, 0 mismatched, 0 failures on either side, 0 missing, identical predictor cache names. `compare` exits 1 only on the provenance flag, explained below | PASS |
| 3 | 16 architecture examples on both trees; `final_20260924/compare_examples.py` | 16 of 16 pass and identical | 16 of 16 pass on each side; 16 of 16 artifact sets identical (`examples_compare.txt`) | PASS |
| 4 | Stage-admission matrix G3b, G9, G10 (`python -m tests.e2e.stage_admission_matrix run` / `compare`) | Every cell PASS and identical | 51 of 51 PASS; the comparator's PASS requires identical `sha256sums.txt` on both path kinds | PASS |
| 5 | W9-05 KV-pressure probe on HEAD (`review_20260924/w9_05_regress/probe.py`) | 72 of 72 drained, 0 short, 0 double-scheduled | 72 of 72 drained, 0 incomplete, 0 short outputs, 0 double-scheduled; 358 decode-phase preemptions, 239 of them while an earlier batch was in flight | PASS |

The provenance flag. Both matrix runs record `source_dirty=true` for all 74
cases. The only change in either worktree is an untracked
`outputs/metrics/meta_llama_llama_2_7b_hf/`, created at 18:19:35Z (base) and
18:19:37Z (head) by the unit suite, which ran first in the same worktree.
`git status` shows no modified tracked file in either tree, and HEAD is exactly
`ba0a804` and `6aee289`. The measured source is therefore the named commits;
the flag records the run order, not a source change.

Commands for 1c (branch worktree, HEAD `6aee289`, submodule `ff11ee6` built,
`PYTHONPATH=$PWD WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp`):

```bash
/data/ycfeng/envs/frontier-py310/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_collective_sim_zero_payload.py
(cd frontier/cc_backend/backends/collective-sim && /data/ycfeng/envs/frontier-py310/bin/python -m pytest -q -p no:cacheprovider tests/test_zero_payload_input.py)
/data/ycfeng/envs/openmopd-py312/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_moe_fused_expert_arithmetic.py tests/unit/test_moe_fused_event_contract.py
```

## 8. Proposals that need a decision

| Proposal | Why it is not done here |
| --- | --- |
| F-R5 victim selection, F-R6 in-flight token | Fidelity changes to preemption; they change results for any preempting run. |
| W3-R5 strict `sync_entry` predicate | 32 regressions in 9 test files; it needs a design choice about which predicate is authoritative. |
| W3-R6 sync-room alias refactor | More than five files in core scheduling code. |
| S44 | Engine-loop semantics shared with S43; proposed as a second row of the S43 task. |
| Native W6 FP8 rerun | **Run 2026-09-24 (user decision A7):** `exp-0924-114241-429126` at `363a1dd`, 8 passed in 14.53 s (`w6_native_20260924/`). It was needed because `f236c17` changed the FP8 step after the last native rerun (`exp-0922-202645-561899` at `c231322`). |
| W7-R4 zero cross-server all-reduce in the companion | **Decided 2026-09-24: option (a), fixed in `e922c77` and gitlink `9adf759`.** Unreachable from Frontier, so no simulation changes. Options were: (a) reject zero again for every kind except all-to-all, restoring the pre-`eb7bc4f` error for them (recommended: it keeps one defined zero case and fails loudly elsewhere); (b) clamp to one byte, as `ff11ee6` does for all-to-all; (c) return 0 ms, as an NCCL call with count 0 does. (b) and (c) define a result no caller needs. The change belongs to companion PR 1 and a gitlink bump. |
| Pinned calibration tools | The E2E normalizer and op-supplement tool must be restored, or a replacement approved, before S43 or S42 can close. |

## 9. Limits

- The first workflow's partial agents did not complete. Each package was reviewed again in the main session, but by one reviewer per package, not by an independent panel.
- Workflow `wf_7606e14e-f10` completed, but 123 of its 152 agents failed on the platform rate limit (section 10). Only its six surviving findings had independent verifiers; the other 32 were checked by the main session alone.
- No GPU run was made during the review itself. The current FP8 code was run natively afterwards (`exp-0924-114241-429126`, 8 passed); FP8 numerics remain compared against no reference.
- The S44 evidence rests on one native case (G4) under dummy Frontier timing.

## 10. Workflow `wf_7606e14e-f10` results (reconciled 2026-09-24)

Workflow `issue26-fix-review-and-scoping`, status `completed`. Result file: the
session's `workflows/wf_7606e14e-f10.json`; per-agent returns in its
`journal.jsonl`. Agents that failed stopped on the platform's weekly limit
(error `rate_limit`, confirmed in an agent transcript).

| Stage | Started | Returned | Failed |
| --- | --- | --- | --- |
| Review, find | 20 | 8 | 12 |
| Review, merge | 6 | 2 | 4 |
| Review, verify | 114 | 18 | 96 |
| Dummy-mode study | 4 | 1 | 3 |
| Scope of S43 and S42 | 8 | 0 | 8 |

### Surviving findings (6)

Each was upheld by three of three verifiers. All six had already been found in
this review:

| Workflow id | Finding | Handled by |
| --- | --- | --- |
| S0 | Zero-payload all-to-all aborts, hangs or costs nothing across servers | W7-R1, companion `ff11ee6`, gitlink `6d621c8` |
| S1 | W7 tests never put a flow into the network simulator | W7-R2, `6d621c8` |
| S2 | Reduce-scatter reachability claim and test | W7-R2, `6d621c8`; reachability corrected in `validation.md` Step 7 and `review.md` W7 |
| S3 | Negative-payload test checks a shared guard | W7-R2, removed in `6d621c8` |
| S4 | Superseded `SyntaxError` guard in the governance scan | W7-R3, `7309f5d` |
| S5 | W9-04 reach recorded for PP=1 co-location only | Records: `issues.md` W9-04 addendum, `validation.md` check A2b. Fix 288 of 288 co-location cells and 36 of 36 PDD cells drain; evidence `wf_7606e14e_f10/w9_04_reach.txt` |

### The 32 findings labelled `refuted`

They were not refuted. The script marks a finding refuted unless at least two
verifiers uphold it, and every verifier for these 32 failed, so each has zero
verdicts. The main session checked each against the code at `5e7221d`:

| Workflow index | Lead | Status |
| --- | --- | --- |
| 0 | Random policy resets the lane per call | W2-R1, `3ec7bbf` |
| 1, 4, 10 | Multi-lane MoE roles reachable through pass-through flags | W2-R5, matrix rows `748e757` |
| 2, 6, 7 | W2 guard tests cannot fail for the defects they name | W2-R3, removed in `3ec7bbf` |
| 3, 5 | The placement rule is written twice | W2-R2, `3ec7bbf` |
| 8 | AGENTS.md calls every role cyclic | W2-R4, `3ec7bbf` |
| 9 | Per-source continuation changes PDD timing | W3-R1, `3a8767c`, `748e757` |
| 11 | `sync_entry` predicate relies on fixture defaults | W3-R5, proposal |
| 12 | Legacy-marker loop | W3-R2, `3a8767c` |
| 13 | Duplicate-owner scan | W3-R3, `3a8767c` |
| 14 | Alias rooms and per-kind partition | W3-R6, proposal |
| 15 | Phase and decoding-request rules duplicated | Phase rule: W3-R4, `3a8767c`. `advance_decode_layer` now says the caller selects the decoding requests; the shared forward filters them and passes `layer_advance_done` to the decode handler. No further change |
| 16, 21 | FP8 config dtype and compute type | W6-R1, W6-R2, `f236c17` |
| 17, 28 | FP8 quantization outside the timed step; `.contiguous()` output copy | W6-R3, `f236c17` |
| 18, 25 | README completeness claims | W6-R5, `b7a7ac5` |
| 19 | Native rerun block shape misquoted | Corrected in the W6 report, section 8, during this review |
| 20 | Parity-test docstring claims it covers the profiler's setup | Docstring corrected in `f236c17`; argument check W6-R6, `6828581` |
| 22 | Nothing checks the profiler's config and alignment arguments | **New:** W6-R6, `6828581` |
| 23 | `block_dims` and `block_shape` encode one quantity | `block_dims` removed in `f236c17` |
| 24 | Import comment promises an impossible fallback | W6-R4, `f236c17` |
| 26 | W6 records contradict each other | **New:** W6-R7, records corrected |
| 27 | Local-reduction test asserts the stub's own definition | Deleted in `f236c17` (commit-message erratum, section 3) |
| 29 | Oracle encodes the superseded idle-engine view | W9-R1, `8ca0387` |
| 30 | No PP>1 case admits into a peer's open group | W9-R2, `6aee289` |
| 31 | Oracle not compared with Frontier; dead publication log | Log removed as W9-R3 (`8ca0387`); **new:** W9-R4, S9-08 qualified |

### Dummy-mode study

One of four parts returned. Its results changed the case records, not the
answer in section 5:
- The case's semantic labels cover the bursts only; the steady segment is unclassified (section 5 row).
- WG04 and WG06 hold under dummy timing only.
- The G4 ground truth ran in eager mode.

These are recorded as errata in `calibration/dp_pp_case_001/analysis/workflow_gap_summary.md` and `semantic_alignment_summary.md` (rows S10 and S39), and as scope rules and blockers in the S43 and S42 plans.

### Scope of S43 and S42

No agent returned. The task records under `/data/ycfeng/Frontier/task_memory/task_2026-09-24_s43_pp_empty_schedule_admission/` and `task_2026-09-24_s42_dp_wave_idle_forward/` were written by the main session. Their plans now also require classifying the steady segment and a vLLM routing record (S39 is `UNSET`) before either task can close.
