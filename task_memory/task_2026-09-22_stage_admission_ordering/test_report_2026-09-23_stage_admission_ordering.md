# Test report — stage admission ordering (P0–P3, P5)

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-23 | R-10 round-2 remediation: §8 added (rule refactor, PDD/online/PD-AF cells, tool fixes); §5.2 and §5.3 restated for the N rows and the barrier-aligned M5. |
| 2026-09-23 | D-9 adopted ("采纳你的推荐，继续"): C3 witnesses judged by co-execution fraction, V5 gated on MoE only. Both comparisons rerun (`aeeca93`); all criteria pass. |
| 2026-09-23 | Created. P0–P3 and P5 executed; two plan stop conditions reached (C3 witness metric at `attn_dp=4`, C7 V5 on the dense shape). P4 push held for the user's decision. |

## 1. Result

| Criterion | Result | Section |
| --- | --- | --- |
| C1 repaired liveness | PASS: all 10 G3a `admission_deadlock` cases complete with conservation; so do the 6 G3b and 2 G7 MoE deadlocks. | §4.1 |
| C2 unchanged controls | PASS: 50 of 50 U cases byte-identical. | §4.2 |
| C3 timing change | PASS (D-9): 6 T cases identical, 8 differ. All 8 differences are start times only (same batches, same component durations), with no self-overlap and `peak_lanes ≤ attn_dp`. All 4 contention witnesses have a strictly larger co-execution fraction. The first comparison stopped on the original absolute-overlap rule; see §4.3. | §4.3 |
| C4 existing tests | PASS: no base-passed node regresses, no new failure or error, skips and collection errors unchanged. | §4.4 |
| C5 rule shape | PASS by review: one predicate, docstrings state the contract, no flag, field, fallback, wake-up, PP branch, second queue or capacity-1 case. | §3 |
| C6 Step 9 probe | Informational: MoE `attn_dp=2, moe_ep=2, PP=2` completes 6/6 (base drains). `PP=3` stops on the known W9-02 node-size rejection. | §4.5 |
| C7 vLLM comparison | PASS (D-9): 50 rows MATCH, 0 MISMATCH, 2 INFORMATIONAL (dense V5). MoE matches on all 26 rows, V5 included; dense matches V1–V4 in every round. The negative controls fail on the base as planned. The first comparison stopped on dense V5; see §5.3. | §5 |

Round 2 (R-10, §8): the rule refactor keeps all 98 cases byte-identical;
50 new PDD, online and PD-AF cells pass (12 more base deadlocks repaired,
0 STOP); G2 shows no regression; the vLLM comparison rerun passes with the
negative controls as separate rows that hold.

Observed facts are separated from inferences. Inferences are marked
"Inference".

## 2. Environment and commits

| Item | Value |
| --- | --- |
| Host | `kun-workspace-vgen2` (CPU) |
| Interpreter | `/data/ycfeng/envs/frontier-py310/bin/python`, Python 3.10.6; distribution digest `ecd50ea8…1902620` for both sets |
| Environment | `PYTHONPATH=<worktree>`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, `TMPDIR=/data/ycfeng/tmp/stage_admission_ordering/pytest-tmp` |
| Base set `base` | run at `a054d87` (harness only; `frontier/` identical to `1f694f7`) |
| After set `after` | run at `dac4e69`, tree clean outside `task_memory/`; 98 cases in 90 s with `--jobs 8` |
| Rule commit | `dac4e69` fix(scheduler): order full-stage admission only behind queued EP waves |
| Harness commits | `a054d87`, `5ade853` (matrix), `799ccb4` (vLLM comparison), `a1b9819` (recorded overlay patch), `aeeca93` (D-9 witness and V5 rules) |
| Scratch root | `/data/ycfeng/tmp/stage_admission_ordering/{base,after,base-rerun,base-pytest,after-pytest,step9_probe}` |

Commands:

```bash
python -m tests.e2e.stage_admission_matrix run --set after --jobs 8
python -m tests.e2e.stage_admission_matrix compare --before base --after after \
  --output /data/ycfeng/tmp/stage_admission_ordering/compare_base_after_d9.json   # first run: compare_base_after.json
python -m pytest tests/<suite> -q -p no:cacheprovider --continue-on-collection-errors \
  --junitxml=<out>/<suite>.xml        # suite in {unit, integration}, base and after
python task_memory/.../evidence/explain_t_path.py <root> <compare json> <out json>
python -m tests.comparison.stage_admission_pp.compare_lanes \
  --vllm-run calibration/stage_admission_case_001/runs/vllm-instrumented/sa-pp-20260923b \
  --before base --after after --output calibration/stage_admission_case_001/analysis
```

## 3. P1 rule

`StageExecutionContext.try_acquire` (`frontier/scheduler/replica_stage_scheduler/stage_execution_context.py`):
an EP wave must be the FIFO head; a full-stage ticket is refused only by an EP
wave queued ahead of it; the admitted ticket leaves the FIFO by
`remove(ticket)`. `_validate_ticket` already rejects a ticket that is neither
queued nor active, so the scan always finds the ticket or an earlier wave.
One file, +21/−7 lines. P1 acceptance: `tests/unit/test_stage_execution_context.py`
and `tests/unit/test_shared_forward_group_admission.py` gave 34 passed with no
assertion change.

## 4. P2 and P3

### 4.0 P2 tests and base negative controls

The new tests were copied into a `git archive 799ccb4` export (rule as on
`1f694f7`) and run there; the log is `evidence/base_negative_controls.log`.

| Test | Expected on base | Observed on base | After P1 |
| --- | --- | --- | --- |
| (a) `test_full_stage_ticket_passes_queued_full_stage_work_but_not_a_queued_wave` | fails at first assertion | fails at line 111, `try_acquire(full1)` is False | pass |
| (a) `test_queued_ep_wave_orders_full_stage_work_on_both_sides` | pass | pass | pass |
| (a) `test_idle_single_owner_stage_admits_a_later_queued_full_stage_ticket` | fails | fails at line 141 | pass |
| (a′) `test_decode_ffn_dense_groups_keep_counter_order_around_a_queued_ep_wave` | pass | pass | pass |
| (b) `test_idle_lane_is_admitted_behind_a_busy_lane_queued_ticket[0,1]` | fails at the other lane's first admission | both fail at line 71, `pop_batch_if_not_busy()` is None | pass |
| (c) `test_moe_lanes_complete_every_request[G3a-moe-dp2-pp2-n4, G3a-moe-dp4-pp2-n8]` | `admission_deadlock` | both `admission_deadlock` | pass: (4, 64, 4) and (8, 128, 8) |
| (c) `test_dense_lanes_start_in_the_same_first_forward` | fails only the same-start assertion | fails `0.05 == 0.0`; lane 1 runs `[0, 0.05]`, lane 0 starts at `0.05` | pass: both lanes start at 0.0 |

### 4.1 C1 — path L (18 cases, all PASS)

Observed `(requests, prefill tokens, decode tokens)` after P1 equals the
generated workload in every case.

| Cases | Observed |
| --- | --- |
| G3a `dp2-pp{2,3}-n{4,8,12}`, `dp4-pp{2,3}-n{8,12}` (10) | n4: (4, 64, 4); n8: (8, 128, 8); n12: (12, 192, 12) |
| G3b `dp2-pp{2,3}-n{4,8}`, `dp4-pp{2,3}-n8` (6) | n4: (4, 64, 12); n8: (8, 128, 24); no mixed-phase failure |
| G7 MoE `dp2-pp2-n{8,16}` (2) | (8, 2048, 8); (16, 4096, 16) |

R0 (informational): the three base deadlocks `moe-dp2-pp2-n4`, `moe-dp2-pp2-n6`,
`moe-dp4-pp2-n8` now succeed; `moe-dp2-pp3-n6` remains
`configuration_rejection` (node-size rule, D-6); the other 12 stay `success`.

### 4.2 C2 — path U (50 cases, all PASS)

Byte-identical `sha256sums.txt`: G1 30 recipes (10 PD-AF included), `PP=1`
cells of G3a (6), G3b (4) and G4 (4, `attn_dp=4, PP=1` included), G5 6.

### 4.3 C3 — path T (14 cases)

Identical hashes (6): G3a/G3b/G4 `dp4-pp{2,3}-n4` (one batch per lane).

Differing (8). `evidence/explain_t_path.py` checks, per stage and lane, that
the ordered batch list, the forward duration and the `execution_time`
component ledger are equal before and after; output
`evidence/p3_t_path_explanation.json`. All 8: `same_batches_and_component_durations = true`,
no self-overlap, `peak_lanes ≤ attn_dp`. Differing files are the ledger,
`request_metrics.csv` and `system_metrics.json` only. Stage 0:

| Case | W | multi-lane time before → after | co-execution fraction before → after | peak lanes | first stage-0 starts before → after | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| G4-dense-dp2-pp2-n4 | | 0.25 → 0.30 | 0.714 → 1.0 | 2 → 2 | {1: 0, 0: 0.05} → all 0 | EXPLAIN |
| G4-dense-dp2-pp2-n8 | W | 0.45 → 0.50 | 0.818 → 1.0 | 2 → 2 | {1: 0, 0: 0.05} → all 0 | EXPLAIN |
| G4-dense-dp2-pp3-n4 | | 0.108 → 0.216 | 0.333 → 1.0 | 2 → 2 | {1: 0, 0: 0.036} → all 0 | EXPLAIN |
| G4-dense-dp2-pp3-n8 | W | 0.216 → 0.396 | 0.375 → 1.0 | 2 → 2 | {1: 0, 0: 0.072} → all 0 | EXPLAIN |
| G4-dense-dp4-pp2-n8 | W | 0.55 → 0.30 | 0.846 → 1.0 | 2 → 4 | {1: 0, 2: 0.05, 3: 0.10, 0: 0.15} → all 0 | EXPLAIN (first run: STOP) |
| G4-dense-dp4-pp3-n8 | W | 0.396 → 0.216 | 0.846 → 1.0 | 2 → 4 | {1: 0, 2: 0.036, 3: 0.072, 0: 0.108} → all 0 | EXPLAIN (first run: STOP) |
| G7-dense-dp2-pp2-n8 | | 0.36 → 0.48 | 0.60 → 1.0 | 2 → 2 | {1: 0, 0: 0.12} → all 0 | EXPLAIN |
| G7-dense-dp2-pp2-n16 | | 0.84 → 0.96 | 0.778 → 1.0 | 2 → 2 | {1: 0, 0: 0.12} → all 0 | EXPLAIN |

Why the two witnesses fail the stated rule (observed from the ledgers): at
`attn_dp=4` the base admits the lanes two at a time; after P1 all four lanes
start every forward together. The stage busy period shrinks from 0.65 to 0.30
(PP=2) and from 0.468 to 0.216 (PP=3), so the time with two or more lanes busy
shrinks with it, although it is now the whole busy period. Request E2E for
`dp4-pp2-n8` drops from 500–700 ms to 300–350 ms with identical batches.
Inference: absolute `multi_lane_busy_time` measures overlap only while the
busy period stays the same length; it cannot express "more overlap" when the
fix compresses the timeline, which happens whenever the base serialized more
than two lanes. The first comparison stopped here with nothing adjusted.
Under D-9 the witness condition is the co-execution fraction, which strictly
increases in all four witnesses (0.818, 0.375, 0.846, 0.846 → 1.0); the rerun
(`compare_base_after_d9.json`) gives U 50 PASS, L 18 PASS, T 6 PASS and
8 EXPLAIN, and no STOP.

### 4.4 C4 — G2 test identities

| Suite | Base | After | Regressions | New failures | Skip / collection changes | Only after |
| --- | --- | --- | --- | --- | --- | --- |
| `tests/unit` | 84 failed, 3644 passed, 49 skipped, 10 errors | 84 failed, 3650 passed, 49 skipped, 10 errors | 0 | 0 | none; `ERROR` lines identical | the 6 new P2 unit tests, all passed |
| `tests/integration` | 11 passed, 21 skipped, 5 errors | 14 passed, 21 skipped, 5 errors | 0 | 0 | none; `ERROR` lines identical | the 3 new P2(c) tests, all passed |

Evidence: `evidence/g2_unit_compare.json`, `evidence/g2_integration_compare.json`;
junit XML under the scratch root.

### 4.5 C6 — Step 9 boundary probe

The original `probe_main.py` wraps `BaseClusterScheduler.on_replica_batch_end`,
a seam that exists only on the PR 35 branch, so it raises `AttributeError`
on this branch. `evidence/step9_probe/probe_completion.py` reuses its
`build_config` unchanged (a100, 6 requests, 16/3 tokens, Poisson) and reports
completion, one process per shape.

| Shape | Base | After |
| --- | --- | --- |
| MoE `attn_dp=2, moe_ep=2, PP=1` | — | 6/6 |
| MoE `attn_dp=2, moe_ep=2, PP=2` | drain, "Sequential simulation ended with non-empty scheduler state" | 6/6 |
| MoE `attn_dp=2, moe_ep=2, PP=3` | — | `ValueError`: collective-sim node-size rule (W9-02, unchanged) |
| dense `attn_dp=1, PP=2` | — | 6/6 |

Composition with PR 35 W3 stays a parent-task check after merge-forward.

## 5. C7 — vLLM comparison (P5)

### 5.1 Ground-truth runs

| Run | RJob | Result |
| --- | --- | --- |
| `sa-pp-20260923a` | `exp-0923-022226-151935`, codesign, 4×H800, creator `i-fengyicheng`, NFS `100.96.128.195:/data/ycfeng/Frontier` | dense complete; MoE failed in `profile_run`: `_moe_C::topk_softmax() expected at most 4 argument(s) but received 5`. Job `Failed`. |
| `sa-pp-20260923b` | `exp-0923-024146-345158`, same shape and mount | MoE and dense complete; job `Succeeded`; worker status 0 |

Cause of the run-a failure (observed): fork commit `1109c4f16` changed
`vllm/_custom_ops.py::topk_softmax` and the `vllm_topk_softmax` call in
`fused_moe.py` to pass a fifth `renormalize` argument, but the fork's own
`csrc/moe/torch_bindings.cpp` (unchanged from `upstream-v0.10.2`) and the
v0.10.2 image both declare the four-argument op. The user decided on
2026-09-23: "topk_softmax 统一修复为4 个参数的版本". Run b applies
`inputs/groundtruth_overlay.patch` (SHA-256 `8d476789…3a9c81`) to the accepted
overlay: it restores the upstream four-argument wrapper and call. The worker
records `_custom_ops.py` as byte-identical to the image's after the patch.
Numerics are unchanged: `vllm_topk_softmax` renormalizes in Python after the
call in both versions. The checkout `494b9f327` is not modified.

vLLM run b: 7 rounds per model (1 warmup + 2 bursts × 3), 152 `pp_boundary` records per
model, no preemption, placement records for every request with none
misplaced; `num_gpu_blocks` 600666 (MoE) and 304854 (dense).

### 5.2 Workflow-gap table (run b)

`analysis/workflow_gap_table.csv`, `analysis/lane_metrics.json`,
`analysis/workflow_gap_status.json`.

| Check | MoE n8 | MoE n16 | Dense n8 | Dense n16 |
| --- | --- | --- | --- | --- |
| V1 completion | 3/3 MATCH | 3/3 MATCH | 3/3 MATCH | 3/3 MATCH |
| N1 / N4 base control (R-10) | N1 HOLDS: base `admission_deadlock` | N1 HOLDS: base `admission_deadlock` | N4 HOLDS: base co-start 1.0 | N4 HOLDS: base co-start 1.0 |
| V2 lane sequences | 3/3 MATCH | 3/3 MATCH | 3/3 MATCH | 3/3 MATCH |
| V3 stage-0 pairing | 3/3 MATCH | 3/3 MATCH | 3/3 MATCH; base pairs 0↔3, 2↔5, …, 6↔none | 3/3 MATCH; base shifted by one forward |
| V4 co-start (vLLM / after / base) | 0.009–0.063 / 0.0 / — | 0.005–0.024 / 0.0 / — | 0.008–0.248 / 0.0 / 1.0 | 0.046–0.171 / 0.0 / 1.0 |
| V5 co-execution (vLLM mean / after / base) | 0.976 / 1.0 / — MATCH | 0.948 / 1.0 / — MATCH | 0.706 / 1.0 / 0.600 INFORMATIONAL (first run: MISMATCH) | 0.865 / 1.0 / 0.778 INFORMATIONAL (first run: MISMATCH) |

Run a (dense only, same scripts): V1–V4 all MATCH; V5 vLLM mean 0.714 (n8)
and 0.685 (n16): MISMATCH under the first rule, INFORMATIONAL under D-9.

Until R-10 the base controls were folded into V1 and dense V4, so a base
without the defect would have turned those vLLM rows into MISMATCH (R2-05).
The rerun at R-10 (`--after after-r2`) writes them as rows N1 and N4: 56 rows,
50 MATCH, 2 INFORMATIONAL, 4 HOLDS, 0 MISMATCH; status PASS,
`negative_control_holds = true`, placement ok with 0 unseen requests.

### 5.3 V5 on the dense shape

`evidence/decompose_co_execution.py` splits the stage-0 non-overlap of each
pair of overlapping forwards into `|Δstart| + |Δend|`
(`analysis/co_execution_decomposition_sa-pp-20260923{a,b}.json`). The identity
holds only for overlapping pairs; at R-10 the script counts disjoint pairs and
checks its pairing against M3. In every round of runs a and b there is no
disjoint pair and no unpaired forward, and the pairing equals M3. The last
column sets both starts of each pair to the later one (R-10, R2-02): vLLM
0.10.2 without CUDA graphs runs the per-forward DP metadata all-reduce inside
`set_forward_context`, after `forward_start_ts`, so neither rank computes
before the later one arrives. The traces carry no timestamp after that
exchange, so this column is derived, not measured.

| Shape (run b) | vLLM M5 per round | Σ start offsets (ms) | Σ end offsets (ms) | stage-0 duration median (ms), CV | M5, starts aligned to the later one (derived) |
| --- | --- | --- | --- | --- | --- |
| MoE n8 | 0.977, 0.974, 0.977 | 0.32–0.54 | 0.19–0.20 | 5.3–8.5, 0.07–0.09 | 0.991, 0.991, 0.994 |
| MoE n16 | 0.978, 0.937, 0.928 | 0.70–3.83 | 0.26–0.69 | 5.3–7.5, 0.05–0.07 | 0.994, 0.988, 0.989 |
| Dense n8 | 0.657, 0.851, 0.609 | 1.49–2.63 | 0.26–4.40 | 3.0–3.8, 0.13–0.29 | 0.722, 0.977, 0.713 |
| Dense n16 | 0.926, 0.833, 0.837 | 1.02–3.60 | 0.60–2.81 | 2.6–2.7, 0.10–0.19 | 0.970, 0.972, 0.883 |

Run a, dense, same columns: observed 0.642, 0.739, 0.760 (n8) and 0.537,
0.752, 0.767 (n16); starts aligned 0.739, 0.843, 0.782 and 0.656, 0.945,
0.950.

Observed:

- vLLM's dense M5 varies between rounds more than the V5 bound: 0.537–0.926
  across the 12 dense rounds of runs a and b. The n16 means of the two runs
  differ by 0.18.
- In every vLLM round the pairing (V3) and the one-to-one lane sequences (V2)
  match the after revision, and the first forwards co-start (V4).
- The non-overlap consists of per-pair start offsets and end offsets. Start
  offsets exceed end offsets in 3 of the 6 dense rounds of run b.
- With both starts aligned to the later one, M5 rises in every round (dense
  0.66–0.98, MoE 0.988–0.994). What remains in dense is the end offsets.
- MoE ends align within 0.2–0.7 ms in total.

Inference (restated at R-10, R2-02): in MoE the EP collectives inside each
forward hold the two ranks together, so vLLM's co-execution is close to
Frontier's 1.0. The dense ranks meet once per forward, in the DP all-reduce.
The dense non-overlap has two sources. First, the rank that reaches the
all-reduce first records its wait as busy time, because `forward_start_ts`
precedes the exchange. Second, the host-bound forwards of about 3 ms vary in
duration per rank. Neither is an admission difference: both ranks enter the
same forward, which is what V1–V4 measure, and they match. The dummy
predictor models neither the wait nor the variation, so Frontier's
co-execution is exactly 1.0 whenever the lanes co-start. The first version of
this paragraph named only the duration variation; the start part was there
too. The dense base (0.600,
0.778) is numerically closer to vLLM only because base serialization removes
overlap; its pairing (V3) and co-start (V4) are wrong in every round.

`compare_lanes.py` labels every `MISMATCH` with the admission owner
`stage_execution_context.py`; on the evidence above, these two rows belong to
the execution-time model instead. The first comparison stopped here with
nothing adjusted. Under D-9 dense V5 is reported, not gated; the rerun gives
`workflow_gap_status.json` status PASS with 0 mismatches. The P5a synthetic
check, now `tests/unit/test_stage_admission_pp_tools.py` (R-10), still flags
a planted late-lane dense round through V3 and V4.

## 6. Decisions

Both stops were resolved by D-9 (`plan.md`), adopted by the user on
2026-09-23 ("采纳你的推荐，继续"):

1. C3: a contention witness passes on a strictly larger co-execution fraction
   `multi_lane_busy_time / busy_time`; the self-overlap and `peak_lanes` checks
   are unchanged.
2. C7: V5 gates the MoE shape only; the dense value is reported with the
   decomposition of §5.3. C7 rests on V1–V4 for both models, V5 for MoE, and the
   base negative controls.

## 7. Verification limits

- The Frontier side runs the dummy predictor; no latency or duration
  parity is claimed (D-8). Dense co-execution against vLLM is therefore not a
  gate (D-9).
- vLLM instrumented mode synchronizes after each forward; stage-1 intervals
  use a wall/monotonic offset and are informational.
- The vLLM ground truth runs with one recorded overlay patch (§5.1); the fork
  checkout still carries the five-argument call and its fork test
  `tests/model_executor/test_enabled_custom_ops.py::test_topk_softmax_wrapper_forwards_renormalize`.
- C6 was measured with a completion-only probe because the boundary seam is
  on PR 35; the PR 35 composition check is pending in the parent task.
- Round 2 (§8): on this branch, online Poisson arrivals reach only lane 0 of
  MONOLITHIC and PREFILL contexts (PR 35's W2 defect on `main`); online
  multi-lane coverage here comes from the `-burst` cells. The derived
  barrier-aligned M5 of §5.3 is not a measurement.

## 8. Round-2 remediation (R-10)

Scope: `plan.md` §7; findings in `review.md` Round 2. Owner instruction:
"确认，执行上上述修复； R2-02 采纳你的推荐；R2-03需要补充  PDD+online（如果你认为pd-af+online有必要，请一并补充）".

### 8.1 Commits and commands

| Commit | Content |
| --- | --- |
| `1661bf1` | R2-01, R2-11, R2-13: `try_acquire` refactor, class docstring, unit test |
| `a8e8d8a` | R2-03, R2-07, R2-08: matrix groups G8–G11, cluster-keyed drain report, case timeout, set lock |
| `e35242f` | R2-02, R2-04, R2-05, R2-09, R2-10, R2-14, R2-15: comparison tools, tool unit tests, evidence scripts |

```bash
# base for the new groups at a8e8d8a, with stage_execution_context.py replaced by
# its 1f694f7 version for the run (run.json: status "M frontier/.../stage_execution_context.py"), then restored
python -m tests.e2e.stage_admission_matrix run --set base --group G8 --group G9 --group G10 --group G11 --jobs 16
python -m tests.e2e.stage_admission_matrix run --set after-r2 --jobs 16     # a8e8d8a, clean outside task_memory
python -m tests.e2e.stage_admission_matrix compare --before base --after after-r2 \
  --output /data/ycfeng/tmp/stage_admission_ordering/compare_base_after-r2.json
python task_memory/.../evidence/explain_t_path.py <root> after-r2 <root>/compare_base_after-r2.json \
  task_memory/.../evidence/r2_t_path_explanation.json
python -m pytest tests/<suite> -q -p no:cacheprovider --continue-on-collection-errors \
  --junitxml=<root>/after-r2-pytest/<suite>.xml
python -m tests.comparison.stage_admission_pp.compare_lanes \
  --vllm-run calibration/stage_admission_case_001/runs/vllm-instrumented/sa-pp-20260923b \
  --before base --after after-r2 --output calibration/stage_admission_case_001/analysis
python task_memory/.../evidence/decompose_co_execution.py <run a> dense    # and <run b> moe dense
```

Environment as §2; interpreter digest `ecd50ea8…` for every set.

### 8.2 Results per finding

| Finding | Check | Expected | Observed | Result |
| --- | --- | --- | --- | --- |
| R2-01, R2-11 | `try_acquire` on an active full-stage ticket, capacity 2 | `False`, context unchanged | base rule `False`; `dac4e69` raises `ValueError` ("not in deque"); `1661bf1` `False`, ticket still active, FIFO unchanged | PASS |
| R2-01, R2-11 | the three context unit files | pass, no assertion change | 181 passed | PASS |
| R2-01, R2-11 | set `after` vs `after-r2`, 98 cases | byte-identical | 97 success hash files identical; the configuration rejection has the same error (`identity_after_after-r2.json`) | PASS |
| R2-03 | groups G8–G11, 50 cases (§8.3) | §4.4 paths | U 18 PASS, L 12 PASS, T 16: 12 PASS, 4 EXPLAIN; 0 STOP | PASS |
| R2-07 | `run --case-timeout 2` on a recipe case | `other_failure`, set completes | `"case timeout after 2 s"` after 2 s; no child or simulator process left | PASS |
| R2-08 | a second `run` while one is running | fails at once | `RuntimeError: another set is running under …; sets share work/ and run one at a time`, exit 1 | PASS |
| R2-04, R2-05, R2-09, R2-10, R2-15 | `tests/unit/test_stage_admission_pp_tools.py` | pass; the ecff89a tools fail the new checks | 9 passed; on the ecff89a tools 7 failed, 2 passed (the late-lane round and the unexpected-file rejection, which the old tools already handled) (`evidence/r2_tool_tests_on_ecff89a.txt`) | PASS |
| R2-05 | `compare_lanes` rerun on run b vs `after-r2` | PASS with controls holding | 56 rows: 50 MATCH, 2 INFORMATIONAL, 4 HOLDS, 0 MISMATCH; `negative_control_holds = true` (§5.2) | PASS |
| R2-02 | decomposition rerun on runs a and b | identity stated for overlapping pairs; derived aligned M5 | no disjoint or unpaired forward in any round; pairing equals M3; existing fields unchanged; aligned M5 in §5.3 | done |
| R2-14 | `probe_completion.py` with `PYTHONPATH` only | C6 shapes complete | `moe_dp2_pp2` 6/6, `dense_dp1_pp2` 6/6; the probe's resolved config and `R0-moe-dp2-pp2-n6`'s differ only in `metrics_config` | PASS |
| G2 | `tests/unit`, `tests/integration` vs `base-pytest` | no regression | unit 84 failed / 3660 passed / 49 skipped / 10 errors (base 84 / 3644 / 49 / 10); integration 14 passed / 21 skipped / 5 errors (base 11 / 21 / 5). 0 regressions, 0 new failures, 0 skip changes; new node ids only: 16 unit, 3 integration (`evidence/r2_g2_*_compare.json`) | PASS |
| R2-12, R2-13 | documentation | contract stated | `design.md` round-2 note; PR body | done |
| R2-06 | pre-merge step P6 | recorded, not executed | `plan.md` §7 | open |

### 8.3 New groups (R2-03)

| Group | Cases | Base | `after-r2` | Paths |
| --- | --- | --- | --- | --- |
| G8 PDD offline | MoE `attn_dp ∈ {2,4}` × `PP ∈ {1,2,3}`, dense `dp1-pp2`, `n=8`, prefill 16 / decode 3 | 4 `admission_deadlock` (MoE `PP > 1`), 3 success | 7 success | U 3 PASS; L 4 PASS |
| G9 PDD online | the G8 shapes at Poisson 20/s; MoE `dp{2,4}-pp{2,3}` burst | 4 `admission_deadlock` (the burst cells), 7 success | 11 success | U 3 PASS; T 4 PASS, identical; L 4 PASS |
| G10 co-location online | MoE (prefill-only) and dense, `dp{2,4}-pp{1,2,3}`, Poisson 20/s and burst; `dp2-pp2` at 5/s and 80/s | 4 `admission_deadlock` (MoE burst `PP > 1`), 24 success | 28 success | U 8 PASS; T 12 PASS, identical, 4 EXPLAIN (dense burst `PP > 1`); L 4 PASS |
| G11 PD-AF | dense and MoE recipes, offline and online, `PREFILL_PP=2` | 4 success | 4 success | U 4 PASS |

Every L case conserves requests and tokens (8 requests, 128 prefill tokens;
24 decode tokens for PD cells, 8 for prefill-only cells). The 4 EXPLAIN cases
(`evidence/r2_t_path_explanation.json`) run the same batches with the same
component durations as the base; only start times differ. Their co-execution
fraction goes from 0.818, 0.375, 0.846 and 0.846 to 1.0, `peak_lanes ≤
attn_dp`, and no lane overlaps itself. These are the same values as the
offline G4 cells of the same shapes.

Observed while running: in the Poisson online cells, every MONOLITHIC and
PREFILL forward runs on lane 0. PDD DECODE lanes all run. `_schedule_batch_mode`
numbers lanes from 0 within each scheduling call, and online arrivals come one
per call. That is PR 35's W2 defect on `main` (fixed on PR 35, not here). The
burst cells deliver all requests at `t=0` in online mode and reach every lane.
They carry the online L and T coverage on this branch. The Poisson cells show
that a multi-lane context with one live lane is unchanged.

PD-AF: `DECODE_ATTN` requires `attn_dp = 1`, and PD-AF has one `attn_dp` for
every role, so no PD-AF context has more than one lane. The G11 cells are
capacity-1 `PREFILL` contexts at `PP = 2`, where R2-12's contract change
applies. They are byte-identical, offline and online.
