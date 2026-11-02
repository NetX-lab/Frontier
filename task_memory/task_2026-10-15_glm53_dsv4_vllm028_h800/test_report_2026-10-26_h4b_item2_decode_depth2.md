# Test report 2026-10-26: H4b-(b) item 2, DECODE batch queue depth 2 with output placeholders

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-26 | Section 7.1: updated the `native_k1_step.py` digest after its refactor for D-H4B-K1-REPRICE; its output is byte-identical. The repricing itself is in [test_report_2026-10-26_h4b_k1_reprice.md](test_report_2026-10-26_h4b_k1_reprice.md). |
| 2026-10-26 | Added section 7.1: the first D step is the step-cost table's k=1 price, and native lifecycle steps measure that entry 8.76 / 12.53 / 0.77 ms lower. Corrected section 10: PREFILL depth 2 does not explain the P2-32 P-side gap (design section 11). |
| 2026-10-26 | Created: the change and its size, default-path identity, the defects found during implementation, depth-1 identity, the new unit tests, the three depth-2 diagnostics against the design section 10.8 expectations, the part comparison with the three lifecycle runs, persistence, deviations from the design and limitations. |

**Authority.** D-H4B-ITEM2 (requirements item 22), D-H200-MAINLINE-CONTINUE (item 21), D-H4A-CHECKPOINT (item 14) and D-NO-INTERRUPT (item 19). Label DIAGNOSTIC: no result here qualifies a cell, no constant is fitted to native06, and H200 and H800 evidence never qualify each other's cells. Design: [h4b_d_side_model/design.md](h4b_d_side_model/design.md) section 10. No GPU job was used; this session holds 0 H200.

## 1. Change

Commit `82390e10` (parent `9b8aaf17`). A new DECODE role field `decode_replica_scheduler_config_max_concurrent_batches` (flat CLI name `cluster_config_decode_replica_scheduler_config_max_concurrent_batches`) sets the queue capacity of the unified DECODE lanes. The default `None` keeps the base value, so nothing changes unless the field is set. An explicit value on unified DECODE models vLLM's AsyncScheduler: a decoding request whose step is still in flight holds an output placeholder and is scheduled again.

| File | Change | Lines |
| --- | --- | --- |
| `frontier/config/cluster_config.py` | the role field (`omit_if_none`, `mode_dependency` pd-disaggregation) | +10 |
| `frontier/config/cluster_role_config.py` | propagation to the DECODE role | +1 |
| `frontier/scheduler/utils/replica_config.py` | `"max_concurrent_batches"` in `_OVERRIDE_FIELDS` of the unchanged role-override registry | +1 |
| `frontier/config/replica_scheduler_config.py` | help text of `max_concurrent_batches` | +3 / −2 |
| `frontier/config/kv_cache_transfer_config.py` | help text of `nixl_read_completion`, which no longer requires depth 1 | +1 / −1 |
| `frontier/entities/batch.py` | `num_output_placeholders`: the stale check of a batch expects the decode-token index its request reaches once the outputs in flight ahead of it are applied (section 3.1) | +27 / −5 |
| `frontier/scheduler/replica_scheduler/base_replica_scheduler.py` | `_create_batch` passes `num_output_placeholders` to `Batch` | +2 |
| `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` | DECODE validation branch (FCFS, PP1, no speculative decoding, no GDN, exact class); placeholder counts at batch creation; claim calls in both schedule loops; report calls at the batch output pop and the dummy-forward end; continuation and in-flight skips bypassed on placeholder lanes; last-token guard; READ table moved out | +72 / −52 |
| `frontier/scheduler/replica_scheduler/vllm_v1_kv_allocation.py` | admission `max_in_flight_tokens = max_concurrent_batches × max_tokens_in_batch`; the two grouped in-flight refusals lifted on placeholder lanes | +14 / −4 |
| `frontier/events/replica_stage_schedule_event.py` | passes the starting batch to `on_engine_step_start` | +1 / −1 |
| new `frontier/scheduler/replica_scheduler/vllm_v1_nixl_reads.py` | mixin `NixlReadCompletion`: register, claim per schedule pass, post at the claiming batch's start, find at any later step start, report at that step's output | +104 |
| Total | 11 files (1 new) | +236 / −65 |

The design estimate was 8 files and about 140-190 changed lines. The three extra files are `batch.py` and `base_replica_scheduler.py` (section 3.1) and the help text in `kv_cache_transfer_config.py`. `vllm_v1_engine_replica_scheduler.py` has 1,914 lines, below the 2,000-line limit; `vllm_v1_kv_allocation.py` has 1,073.

Tests: new `tests/unit/test_decode_output_placeholders.py` (315 lines, 7 tests); `tests/unit/test_nixl_read_completion.py` (+63 / −6, 9 tests; the depth-1 restriction test is replaced by two depth-2 tests); fixture or expectation edits in 4 existing files (section 5).

## 2. Default path is unchanged (design section 10.9, item 1)

| Check | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| Full unit suite | `/data/ycfeng/tmp/glm53_dsv4/h4b_item2/run_unit_final.sh` (`pytest tests/unit -q -p no:cacheprovider -p no:logging --continue-on-collection-errors -rfE`, frontier-py310, Python 3.10.6) | failing or erroring test IDs equal to the item 1 run (`/data/ycfeng/tmp/glm53_dsv4/h4b_item1/unit_fix.log`: 241 IDs) | 134 failed, 6133 passed, 208 skipped, 108 errors in 367.7 s (item 1: 134 failed, 6122 passed, 208 skipped, 108 errors). The 241 failing or erroring IDs equal item 1's (`cmp_ids.py`: 0 added, 0 removed). `collect_diff.py` compares the collected IDs with the item 1 tree `.worktrees/fidelity-candidate-96d9c1ef`: 6565 against 6554, 13 added and 2 removed (section 5); all 13 added pass | PASS |
| Integration suites and the e2e gate unit test | `run_focus.sh` (`pytest tests/unit/test_e2e_gate.py tests/integration`), and `run_focus_base.sh`, the same command in the clean worktree `.worktrees/fidelity-candidate-96d9c1ef` | the same failures on both trees | both trees: 1 failed, 157 passed, 22 skipped, 5 errors; the same 6 IDs (`cmp_ids.py`: 0 added, 0 removed). They are `test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward`, which already failed before item 1, and 5 errors of `test_pdaf_reference_lifecycle_observer.py`, which need the pinned PD-AF Reference checkout | PASS |
| Dummy smoke matrix | `run_smoke.sh` (`PYTHON_BIN=…/frontier-py310/bin/python bash examples/architecture/run_dummy_smoke_matrix.sh`) | 12 cases, exit 0 | 12 cases completed, rc=0 | PASS |
| 77-case refactor fidelity | `run_fidelity.sh` (`tests/e2e/refactor_fidelity/run_matrix.py run --jobs 8 --clean-cache`, then `compare` against the `96d9c1ef` candidate output `/data/ycfeng/tmp/glm53_dsv4/h4b_item1/fidelity_96d9c1ef/candidate_96d9c1ef`), from the worktree `.worktrees/fidelity-candidate-item2` (`9b8aaf17` plus the item 2 patch) | every output identical | 77 of 77 compared, 77 identical, 0 mismatched, 0 failures on either side, predictor cache names compared and clean. The new field is `omit_if_none`, so `config.json` has no echo of it | PASS |

The fidelity matrix covers GLM, co-location, PDD and PD-AF cases. `compare` exits 1 only because of its provenance finding: the candidate is a modified working tree at `9b8aaf17`, not a commit. The 11 `frontier/` files in that worktree were compared with `cmp` against this worktree and are identical; they are the files committed in `82390e10`. Report: [h4b_d_side_model/results/fidelity_comparison_item2.json](h4b_d_side_model/results/fidelity_comparison_item2.json).

## 3. Defects found during implementation

### 3.1 The second in-flight batch was dropped as stale

**Observed.** An early depth-2 probe in this session logged 28 stale-batch-end warnings: a batch formed while the request's previous step was in flight was dropped when it ended. That probe's log was not kept.

**Cause.** A `Batch` records each request's `current_decode_token_index` when it is formed, and the stale check at its end compares that with the request's index. The second in-flight batch is formed while the first is still in flight. By the time it ends, the first batch's output has advanced the index by one, so the check judged it stale.

**Fix.** `Batch` takes `num_output_placeholders`, the number of the request's steps in flight when the batch is formed. That is vLLM's `num_output_placeholders`. The expected index adds it. Only placeholder lanes pass the counts, so on every other lane the count is 0 and the check is unchanged.
- Evidence: in the probe `/data/ycfeng/tmp/glm53_dsv4/h4b_item2/explore/e1.py`, the two batches of one pass carry the signatures `(0, 0, 0, 1)` and `(0, 0, 0, 2)`. The three depth-2 DSV4 runs log no stale warnings (section 6).
- Deviation: design section 10.6 said "`Batch` is unchanged".

### 3.2 Preemption victim under the priority policy

Design section 10.5 relies on a victim being in at most one in-flight batch. Under FCFS the victim is the running tail, which no earlier part of the same pass scheduled. The priority policy may instead pick a request that the same pass already put in a batch. The existing in-flight sample path does not handle that case. The DECODE validation therefore requires FCFS, which every DSV4 case uses. No new guard was added.

### 3.3 Unreachable DECODE prefix-caching check

The design listed "no prefix caching" in the DECODE validation. The unified DECODE role has no prefix-caching manager, so the check cannot fail. It was dropped.

### 3.4 Flat CLI name

Design section 10.9 used the shorthand `--decode_replica_scheduler_config_max_concurrent_batches`. The H4a-4 driver rejected it as an unknown field. The flat name is `cluster_config_decode_replica_scheduler_config_max_concurrent_batches`.

## 4. Depth-1 engine_step identity (design section 10.9, item 2)

Command: `bash /data/ycfeng/tmp/glm53_dsv4/h4b_item2/run_diag.sh <case> <case>_d1`, the item 1 configuration (`nixl_read_completion=engine_step`, DECODE capacity unset). Comparison: `cmp_d1.py` (SHA-256 per output file, then request rows) and `cmp_cfg.py` (config tree), against the item 1 scratch outputs `h4b_item1/diag/{native06f,p2_32f,p2_64f}`.

| Case | `request_metrics.csv` | `system_metrics.json` | `metrics_ground_truth.jsonl` | `frontier_transfer_ledger.jsonl` | `op_precision_metadata.csv` | `config.json` | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native06 | identical | identical | identical | identical | identical | 3 scratch paths only | PASS |
| P2-32 | identical | identical | identical | identical | identical | 3 scratch paths only | PASS |
| P2-64 | identical | identical | identical | identical | identical | 3 scratch paths only | PASS |

The three `config.json` differences are `cc_backend_config/cache_dir`, `metrics_config/cache_dir` and `metrics_config/output_dir`. Wall time: 768.3 s, 783.1 s and 1,114.6 s.

At depth 1 the new READ rule can differ from item 1's in two situations. Neither occurred in these runs:
- (a) A READ registered at the same simulated time as a schedule pass, after it, before that pass's batch starts. Item 1 posts it at that start; the new rule leaves it to the next pass (design section 10.6).
- (b) A READ found at a dummy-forward start on a DP lane. Item 1 handed every found READ to the next report call site. The new rule reports it at the end of that dummy forward. The two differ when another report call site runs while the dummy forward is in flight.

## 5. Unit tests (design section 10.9, item 3)

All 7 new and 9 updated test functions pass inside the full unit run of section 2. Counted as collected IDs, the change adds 13 and removes 2: the 7 new functions give 11 IDs, because the scope test has 5 parameter cases; `test_nixl_read_completion.py` adds 2 and removes 1; `test_vllm_v1_max_model_len.py` removes its `decode` case. That is the +11 passed of section 2. Evidence: `/data/ycfeng/tmp/glm53_dsv4/h4b_item2/collect_diff.txt`.

`tests/unit/test_decode_output_placeholders.py`:
- `test_decode_role_override_reaches_only_the_decode_scheduler`: DECODE gets capacity 2, PREFILL keeps `None`.
- `test_only_an_explicit_decode_capacity_uses_output_placeholders`.
- `test_explicit_decode_capacity_rejects_unsupported_scopes`: PP2, the priority policy, speculative decoding, a GDN model and a subclass.
- `test_in_flight_request_is_scheduled_again_until_its_last_token_is_in_flight`: contexts 16 / 17 / 18, signatures offset by the placeholders, an empty pass once the last token is in flight.
- `test_preemption_victim_with_a_step_in_flight_still_takes_that_step_sample`.
- `test_dp_lane_with_grouped_allocation_runs_two_steps_of_one_request`: DSV4 grouped allocation on a DP lane, contexts 99 / 100 / 101.
- `test_grouped_admission_bounds_the_tokens_of_both_in_flight_batches`.

`tests/unit/test_nixl_read_completion.py`: the depth-1 restriction test is replaced by `test_depth_two_read_rides_its_pass_batch_and_reports_with_the_finding_batch_output` and `test_pass_that_forms_no_batch_leaves_its_reads_to_the_next_step_of_any_kind`. The other 7 tests keep their names. They gain the claim step that the new rule needs before a READ is posted, and the dummy-forward test also checks that the READ is reported at the end of the dummy forward.

Edits to existing tests:
- `test_vllm_v1_max_model_len.py`: DECODE is no longer in the parametrization of `test_explicit_queue_capacity_rejects_disaggregated_roles`, because unified DECODE now accepts an explicit capacity.
- `test_grouped_kv_cache_scheduler.py`: its hand-built scheduler fixture sets `_max_concurrent_batches = 1` and `_uses_output_placeholders = False`.
- `test_module_split_boundaries.py`: the expected mixin order gains `NixlReadCompletion`.
- `test_pdaf_decode_attn_online_cohort.py`: its hand-built scheduler fixture gains the two new READ tables.

## 6. Depth-2 DSV4 diagnostics (design section 10.9, item 4)

Commands (`/data/ycfeng/envs/frontier-py310/bin/python`, Python 3.10.6), from the repository root:

```bash
# Simulation; ulimit -v 62914560, timeout 21600
bash /data/ycfeng/tmp/glm53_dsv4/h4b_item2/run_diag.sh <case> <case>_d2 cluster_config_decode_replica_scheduler_config_max_concurrent_batches=2
#   = h4a4_driver.py <case> diag/<case>_d2 analytical_kv_cache_transfer_config_nixl_read_completion=engine_step cluster_config_decode_replica_scheduler_config_max_concurrent_batches=2
# Client gate and H4a-5 attribution; ulimit -v 31457280, timeout 1800 each
bash /data/ycfeng/tmp/glm53_dsv4/h4b_item2/run_gate_attr.sh <case> <case>_d2
# Parts against the lifecycle runs
/usr/bin/python3 h4b_d_side_model/results/item2_part_comparison.py <task directory>
```

Run facts:
- Each receipt's `override_diff` adds the DECODE capacity (`None` → 2) to the item 1 overrides. The receipts record source `9b8aaf17` with the 11 `frontier/` paths of section 1 modified.
- A configuration probe on P2-32 (`probe_cap.py`) shows the DECODE lanes at capacity 2 with output placeholders and the PREFILL lanes at capacity 1 without them.
- KV admission is unchanged: 2,940 blocks at `max_model_len` on every DECODE lane against native 2,944, which DIFFERS as before. `max_model_len` = 7,168 binds before the in-flight term.
- Every run has 0 preemptions and no stale-drop warning in its log, and ends with exit 0.
- Wall time: 770.1 s (native06), 793.6 s (P2-32) and 1,104.0 s (P2-64).

Client gate (relative error limit 10%; native is the mean of the three reference repetitions):

| Case | Metric | Native | Item 1 (depth 1) | Item 2 (depth 2) | Change | Design 10.8 expectation | Gate, depth 2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native06 (H800-32) | TTFT (ms) | 398.82 | 365.06 (−8.46%) | 403.06 (+1.06%) | +38.00 | 404.03 (+1.3%) | PASS |
| | TPOT (ms) | 19.009 | 19.485 (+2.50%) | 19.485 (+2.50%) | 0.000 | nearly unchanged | PASS |
| | E2E (ms) | 19,845.5 | 20,298.2 (+2.28%) | 20,336.2 (+2.47%) | +38.00 | — | PASS |
| | THPT (tok/s) | 6,272.1 | 6,237.3 (−0.55%) | 6,235.7 (−0.58%) | −1.64 | — | PASS |
| P2-32 (H200-32) | TTFT (ms) | 364.03 | 361.10 (−0.81%) | 397.26 (+9.13%) | +36.17 | 398.38 (+9.4%) | PASS |
| | TPOT (ms) | 19.919 | 18.642 (−6.41%) | 18.642 (−6.41%) | 0.000 | nearly unchanged | PASS |
| | E2E (ms) | 20,741.1 | 19,431.4 (−6.31%) | 19,467.6 (−6.14%) | +36.17 | — | PASS |
| | THPT (tok/s) | 6,231.4 | 6,277.7 (+0.74%) | 6,276.2 (+0.72%) | −1.58 | — | PASS |
| P2-64 (H200-64) | TTFT (ms) | 741.22 | 647.55 (−12.64%) | 695.45 (−6.17%) | +47.90 | 697.09 (−6.0%) | PASS on the mean, **FAIL** on repetition 1 |
| | TPOT (ms) | 25.871 | 24.768 (−4.26%) | 24.768 (−4.26%) | 0.000 | nearly unchanged | PASS |
| | E2E (ms) | 27,206.8 | 25,985.0 (−4.49%) | 26,032.9 (−4.31%) | +47.90 | — | PASS |
| | THPT (tok/s) | 5,963.2 | 5,999.0 (+0.60%) | 5,997.0 (+0.57%) | −1.94 | — | PASS |

TTFT against each reference repetition:

| Case | Repetition 1 | Repetition 2 | Repetition 3 | Verdict |
| --- | --- | --- | --- | --- |
| native06 | 403.31: −0.06% | 393.22: +2.50% | 399.95: +0.78% | PASS |
| P2-32 | 361.97: +9.75% | 364.75: +8.91% | 365.37: +8.73% | PASS |
| P2-64 | 779.89: **−10.83%** | 725.49: −4.14% | 718.27: −3.18% | FAIL (repetition 1) |

Observed:
- TTFT is within 1.7 ms of the design 10.8 expectation on all three cases: −0.97, −1.11 and −1.64 ms.
- TPOT is unchanged to six decimals on all three cases. E2E rises by exactly the TTFT change, and THPT falls by at most 0.03 points.
- On native06 and P2-32 the gate stays PASS. P2-64 now passes on the mean of the repetitions, and also on repetition 2, which failed at depth 1. It still fails on repetition 1, whose first-contact READs raise its TTFT (P2-64 lifecycle report, section 5).
- The P2-32 error moved from −0.81% to +9.13%. The design predicted this: the P side is 45.92 ms too long, and the D-side handoff no longer offsets it (attribution below).

H4a-5 attribution (ms; gap = Frontier − native mean of repeats):

| Case | Depth | TTFT gap | P-side gap | Handoff gap | Client-edge gap |
| --- | --- | --- | --- | --- | --- |
| native06 | 1 | −33.76 | +12.69 | −45.79 | −0.66 |
| | 2 | +4.24 | +12.69 | −7.79 | −0.66 |
| P2-32 | 1 | −2.93 | +45.92 | −48.24 | −0.61 |
| | 2 | +33.24 | +45.92 | −12.07 | −0.61 |
| P2-64 | 1 | −93.66 | +6.82 | −99.67 | −0.82 |
| | 2 | −45.76 | +6.82 | −51.77 | −0.82 |

Only the handoff gap moves. It closes by 38.00, 36.17 and 47.90 ms; design section 10.8 expected 2 × S_F = 38.97, 37.28 and 49.54 ms.

## 7. Parts against the three lifecycle runs

Tool: [h4b_d_side_model/results/item2_part_comparison.py](h4b_d_side_model/results/item2_part_comparison.py), which reuses the readers of `item1_part_comparison.py`; output [item2_part_comparison.txt](h4b_d_side_model/results/item2_part_comparison.txt). The native parts are the same as in the item 1 report, section 5. For P2-64l they come from repetitions 2-3. S_F is the Frontier TPOT mean.

| Part (ms) | native06 / H800l | P2-32 / P2-32l | P2-64 / P2-64l r2-r3 |
| --- | --- | --- | --- |
| Native b + c + d + e | 79.20 | 91.79 | 104.18 |
| Native f | 41.80 | 42.05 | 51.53 |
| Native a + g | 4.90 | 4.63 | 3.95 |
| Native handoff | 125.91 | 138.47 | 159.66 |
| Frontier depth 1: kv_transfer + d_first_step | 47.70 + 30.01 = 77.71 | 45.57 + 34.11 = 79.68 | 61.34 + 26.76 = 88.10 |
| Frontier depth 2: kv_transfer + d_first_step | 66.70 + 49.00 = 115.70 | 63.66 + 52.19 = 115.85 | 85.29 + 50.71 = 136.00 |
| Depth 2 − depth 1: kv_transfer / d_first_step | +19.00 / +19.00 | +18.08 / +18.08 | +23.95 / +23.95 |
| Depth 2 kv_transfer in S_F | 3.42 | 3.41 | 3.44 |
| Depth 2 d_first_step in S_F | 2.51 | 2.80 | 2.05 |
| Depth 2 − lifecycle: handoff | −10.20 | −22.62 | −23.66 |
| of which kv_transfer − (b + c + d + e) | −12.50 | −28.14 | −18.89 |
| of which d_first_step − f | +7.20 | +10.14 | −0.82 |
| of which − (a + g) | −4.90 | −4.63 | −3.95 |
| Lifecycle run − target (reference mean of repeats) | +2.41 | +10.55 | −28.10 |

Observed:
- Each of the two Frontier parts rose by the same amount, about 0.97 S_F. This matches the design: one step more from P end to the READ report and one step more from the report to the first D token.
- From P end to the report now takes about 3.4 Frontier steps on all three cases, against about 2.45 at depth 1. Design 10.8 expected "d about 3 Frontier steps from the claiming pass". The P-end-to-claim wait adds about half a step to that.
- The first D step is now 2.05-2.80 Frontier steps; the design expected about 2. The excess over the rule is the same as at depth 1 (1.08-1.83 for a rule of 1), so item 2 did not add it. Section 7.1 traces it to the k=1 entry of the step-cost table.
- On native06 and P2-32, Frontier's first D step is now longer than native f (+7.20, +10.14 ms), while the part before the report is shorter than native b + c + d + e (−12.50, −28.14 ms).

Inference about the remaining handoff gap, not measured separately:
- Against the lifecycle runs it is −10.20, −22.62 and −23.66 ms.
- On the READ side, the late READ handles (design section 6.2, estimated at 16.6 ms on H200-32) and the ingress (section 6.3, about 8.6 ms on H200-32) are still not modeled. Their sum is about the P2-32 gap.
- The P2-64 target is another 28.10 ms above its lifecycle run: about 16.5 ms from the repetition 1 first-contact READs and 11.6 ms of run-to-run difference (item 1 report, section 6).

### 7.1 The first D step (added 2026-10-26)

Approved as a CPU-only investigation (D-H4B-AFTER-ITEM2, requirements item 23). Tools, all reading persisted copies:
- [first_d_step.py](h4b_d_side_model/results/first_d_step.py), output [first_d_step.txt](h4b_d_side_model/results/first_d_step.txt): Frontier's first D step per request, from the persisted depth-1 and depth-2 runs, against the DECODE entries of the case's step-cost table;
- [native_k1_step.py](h4b_d_side_model/results/native_k1_step.py), output [native_k1_step.txt](h4b_d_side_model/results/native_k1_step.txt): native D step durations from the scheduler records of the three lifecycle runs. The k=1 step is the step whose schedule promotes a formal request (6143 computed + 1 scheduled token).

```bash
# ulimit -v 31457280, timeout 600-900; Python 3.10.6 (/data/ycfeng/envs/frontier-py310/bin/python)
python h4b_d_side_model/results/first_d_step.py .
python h4b_d_side_model/results/native_k1_step.py .
python h4b_d_side_model/results/native_p_service.py .   # used by design section 11
```

A second run of all three scripts reproduced each `.txt` byte for byte, with empty stderr. SHA-256:

| File | SHA-256 |
| --- | --- |
| `first_d_step.py` | `719771f014a191606457898425e7e26e1973ae6c01ac28c0bc7c0e976eb7799a` |
| `first_d_step.txt` | `65fc58d80555dd30be07b0a14338a8e8d6cba5dfb41daa9512a954f53c9cd07b` |
| `native_k1_step.py` | `7030c117124288702c54ec37d0d27734bfdbc872f086e948571131f12adf9f0c` (refactored for D-H4B-K1-REPRICE; output unchanged; the 7.1 version was `0c35c213…`) |
| `native_k1_step.txt` | `0f77b535892a8120f4755b35dae024d46b73f9a11b595df977e959c9efab7fd1` |
| `native_p_service.py` | `40cbae482cf98de0c4d1be42805150454572193e141f136e824d3283972709b8` |
| `native_p_service.txt` | `60202fd3489441eace5e6a84c18707a5c9c903293599f50a9b979a021064a98b` |

Frontier's first D step against the table (ms):

| Case | Table: k=1 alone | k=1 with decodes | decode-only | Depth 1: p50 (mean) | Expected | Depth 2: p50 (mean) | Expected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native06 | 21.10 | 30.15 | 19.30 | 30.15 (30.01) | 30.15 | 49.45 (49.00) | 49.45 |
| P2-32 | 25.28 | 34.25 | 18.37 | 34.25 (34.11) | 34.25 | 52.62 (52.19) | 52.62 |
| P2-64 | 26.76 | 26.76 | 24.72 | 26.76 (26.76) | 26.76 | 51.48 (50.71) | 51.48 |

Expected values: at depth 1 the k=1 step with decodes; at depth 2 the decode-only step of the batch already in flight, then the k=1 step with decodes.

Observed:
- On all six runs the median first D step equals the expected value. The minimum equals the k=1-alone entry: a request on a lane with no other decoding request. That is why the means are slightly lower.
- So the scheduling adds exactly the designed steps. The excess over 1 or 2 TPOT is the k=1-with-decodes entry being 10.85, 15.88 and 2.04 ms above the decode-only entry.
- That entry comes from 1 traced step on native06 and P2-32, with 1 decode request, in the profiled repetition 1. Its client TPOT is 22.86 and 22.84 ms, against 19.83-19.85 and 19.08-19.09 ms in the unprofiled repetitions. On P2-64 it is the listed substitution from the k=1-alone entry (D-H200-STEP-COST-SUBSTITUTION).

Native k=1 step in the unprofiled lifecycle runs (ms, `native_k1_step.txt`):

| Run | k=1 step with decodes: n, median | Decode-only step: median | Table k=1 with decodes | Table − native |
| --- | --- | --- | --- | --- |
| H800l | 378, 21.39 | 19.83 | 30.15 | +8.76 |
| P2-32l | 378, 21.72 | 20.17 | 34.25 | +12.53 |
| P2-64l | 372, 25.99 | 24.45 | 26.76 | +0.77 |

- Natively the k=1 step with decodes is 1.54-1.56 ms longer than a decode-only step on all three runs.
- The table's excess over the native step (+8.76, +12.53, +0.77 ms) has the sign and about the size of Frontier's d_first_step − f above (+7.20, +10.14, −0.82 ms).
- The 6, 6 and 12 k=1 steps alone take 61-69 ms (median); they are the idle-lane requests and are not analyzed further.

Conclusion: the first-D-step excess is the price of one step-cost table entry, measured from a single profiled step. The scheduling model is not its cause. Whether profiling or the single sample's composition inflates that step is not measured.

Not done: the table is not repriced. Pricing that entry from the lifecycle runs (native, unprofiled, not the comparison target) would change a DSV4 measured input. That needs YC's decision (progress.md, next actions).

## 8. Persistence (D-INPUT-PERSISTENCE)

`persist.py` copied each run from scratch `/data/ycfeng/tmp/glm53_dsv4/h4b_item2/diag/<case>_d2` and its `gate/` and `attr/` outputs, and checked every copy by SHA-256. Each directory holds the same 16 files as item 1: `config.json`, `metrics_ground_truth.jsonl`, `request_metrics.csv`, `system_metrics.json`, `receipt.json`, `run.log`, `h4a5_attribution.json`, `h4a5_attribution.out`, `gate/gate.log`, `gate/matrix.json`, `gate/reading.json` and `gate/rows/`.

| Directory | Scratch name | Files | Bytes |
| --- | --- | --- | --- |
| `h4a_p0_diagnostic/results/run_engine_step_depth2/` | `native06_d2` | 16 | 425,575 |
| `h200_mainline/diagnostic/results/p2_32e2/` | `p2_32_d2` | 16 | 425,447 |
| `h200_mainline/diagnostic/results/p2_64e2/` | `p2_64_d2` | 16 | 426,368 |

Digests:

| File | SHA-256 |
| --- | --- |
| `run_engine_step_depth2/request_metrics.csv` | `2610c783610b5f5b56e4d45060a217f4c15450c98706802bb77a254ca4ec83ac` |
| `run_engine_step_depth2/system_metrics.json` | `4727397f63763863b39017cf2338ac7a84ba50b6891c96312df945575eca7f85` |
| `run_engine_step_depth2/h4a5_attribution.json` | `2b2a463b77030331d537ea90a797fbc889e25c5f96ebb7f8490422292181e9f6` |
| `run_engine_step_depth2/gate/reading.json` | `6e0ba3c1e0a844fa1cce368239988baeef2e9facf84a5babfb127199c78e06cd` |
| `run_engine_step_depth2/receipt.json` | `2157cf7529c0aeda4658a3a6326a01d8c5b9f29064d1215c1cf42ab8aa10c5d7` |
| `p2_32e2/request_metrics.csv` | `04c0785ff7b5d3b437129af4c284cc63b1b053a565882ba8318c2bec5312d3a7` |
| `p2_32e2/system_metrics.json` | `5cc3fbc57b3f5619d1f5b6df18474b0cc05349c5b6b540c7ad85fe12379a5319` |
| `p2_32e2/h4a5_attribution.json` | `7f44d4c571509985d58877f455a636403593c912b77c07c69662421ebbd2dfc4` |
| `p2_32e2/gate/reading.json` | `bf40e8df2380fcd2482aace338193a72f974aa328e9cf158cd5e560ae6e8a6be` |
| `p2_32e2/receipt.json` | `f7cc1fcbb2f70202e30224e2d28db5f52a5ec2f76859a05caf452ed573272cdf` |
| `p2_64e2/request_metrics.csv` | `862d2ed14a096783ed481f3ade9f6148b971fccb8dd5d4725293a25cbfbe1122` |
| `p2_64e2/system_metrics.json` | `e49f3e9d54b90ae2b5d144348e30b0c95f5a4c767ccefd203edf929a54006492` |
| `p2_64e2/h4a5_attribution.json` | `2c0acc08a729a1af69af014f6deff621ce1c9a6c0c04c1034263b1666b214eaf` |
| `p2_64e2/gate/reading.json` | `3a02c4e771b3353da141ef5c6ce8c9f44ae2f3f90e139e99a5b61341dfec60f2` |
| `p2_64e2/receipt.json` | `b527dfbc9c9980d663bd382cd0197a44f01e898004c155a7e53c51d354359ab6` |
| `h4b_d_side_model/results/fidelity_comparison_item2.json` | `43453b006ab3a7896a66f32a5085d4b6bfe433b6bbfc050a4b2e105127042735` |
| `h4b_d_side_model/results/item2_part_comparison.txt` | `0fca856f1063c088d7dd0afc4ea360f994cf60a96217d9316761fe95a27ccf56` |

The depth-1 reruns are byte-identical to the persisted item 1 results (section 4), so they are not persisted again. The scratch runs, logs and the fidelity worktree `.worktrees/fidelity-candidate-item2` are left in place. Removing the worktree needs YC's approval. YC approved the removal (D-H4B-AFTER-ITEM2); it was removed on 2026-10-26 after its tracked files were checked identical to `82390e10` (progress.md).

## 9. Deviations from design section 10

Adopted under D-NO-INTERRUPT; each follows the lead's recommendation.

| Design | Implemented | Reason |
| --- | --- | --- |
| "`Batch` is unchanged" (10.6) | `Batch` and `BaseReplicaScheduler._create_batch` take `num_output_placeholders` | Without it the second in-flight batch is dropped as stale (section 3.1) |
| DECODE validation: no prefix caching (10.2) | No prefix-caching check | Unreachable on unified DECODE (section 3.3) |
| Validation does not mention the scheduling policy (10.2) | FCFS required | A priority victim can sit in two batches (section 3.2) |
| Admission formula only on placeholder lanes (10.4) | Applied on every lane | Every other lane has capacity 1 under grouped allocation, so the value is the same; the 77-case fidelity matrix is identical |
| Field with no `omit_if_none` (10.7) | `omit_if_none` | Keeps `config.json` of every other run unchanged; no config echo in the fidelity matrix |
| 8 `frontier/` files, about 140-190 lines (10.7) | 11 files, +236 / −65 | The three extra files above; the READ table move counts on both sides |
| One possible depth-1 difference: a same-time registration (10.6) | Two possible differences, (a) and (b) | Section 4; neither occurred |
| `--decode_replica_scheduler_config_max_concurrent_batches` (10.9) | `cluster_config_decode_replica_scheduler_config_max_concurrent_batches` | The flat CLI name (section 3.4) |
| — | Fixture or expectation edits in 4 existing test files | Section 5 |

## 10. Limitations

- DIAGNOSTIC only. One Frontier run per case, against one reference run each.
- The deferred block free of vLLM (design 10.1, point 6) is not modeled, so a preempted victim's blocks return at most one step early. The DSV4 runs have no preemption.
- `Request.on_batch_schedule` keeps only the latest schedule time. At depth 2, per-iteration service-time metrics derived from it are understated; TTFT, TPOT and E2E are unaffected.
- PREFILL depth 2 is not modeled; the P engines also run `--async-scheduling`. It is item 3 ([design](h4b_d_side_model/design.md) section 11). It has no effect on the three cases and does not explain the P2-32 P-side +45.92 ms. That gap is the run-to-run difference in native P service between the profile run and the reference run (design section 11.4). Corrected 2026-10-26: this line earlier said item 3 may explain part of the gap.
- The k=1-with-decodes entry of the native06 and P2-32 tables is 8.76 and 12.53 ms above the native lifecycle measurement (section 7.1).
- MONOLITHIC capacity 2 keeps its skip semantics (design 10.2, option C).
- The late READ handles and the ingress are still not modeled. The P2-64 repetition 1 first-contact READs are not modeled either.
- The P2-32 TTFT error of +9.13% comes from the P side.
- Section 7 compares means, not matched requests, and the lifecycle runs are separate runs on other hosts.
