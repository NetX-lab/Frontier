# 06: Native TP/EP/NIXL measurement and narrow cost model

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Recorded both F1b terminal coverage failures; no incomplete-request metric gate or formal input change. |
| 2026-11-01 | Synchronized completed A6/F3 evidence and the active F1/F2 comparisons; acceptance criteria unchanged. |
| 2026-10-31 | Synchronized current Codex/H200 operator evidence and pending work; preserved historical snapshots and qualification requirements. |
| 2026-10-22 | Active communication/coverage acceptance DSV4 only; parked GLM all-tier work and removed GLM-first consumer/completion dependency while retaining measured rows and hardware-specific limits. |
| 2026-10-21 | Accepted54rows/248traces;1241-row generation preserves1187 prior rows; exact P/D lookup0.731449604ms succeeds. |
| 2026-10-21 | Bound216 audited EP16 rows; submitted exact SP768/1 supplement exp-1021-214338-130275 on16H800 with per-rank Reduce/root traces after source/image acceptance. |
| 2026-10-21 | Communication now follows each priority hardware/topology, with H80032 first and reuse of matching EP16/EP32 measurements before supplementation. |
| 2026-10-21 | Retargeted H200 collective/NIXL payload coverage to 128 fixed 6144/1024-token requests at QPS 1.0, no chunking and contexts through total length 7168; preserved measured-backend typed coverage and historical H800 evidence. |
| 2026-10-21 | Narrowed to communication cost for the canonical case: collective curves and NIXL readings are re-collected on H200 at the frozen topology's groups. The measured backend stays in use for the formal case (Q5=B), so a missing H200 shape is a typed coverage error that stops the run rather than an extrapolation. |
| 2026-10-20 | Handoff status: GLM raw measurements and K5 re-measurement done on H800; curated curves built; K1-K4 approved, not implemented. |
| 2026-10-20 | raw_measurements[GLM] unblocked by G0-B closure; current workstream W5. |
| 2026-10-18 | Documentation review: placed target curves after represented paths and split lifecycle from measured contention to avoid over-coupling. |
| 2026-10-18 | Completed existing-curve consumer source preparation and3CPU checks; corrected one reviewed report-only fallback statement without changing a backend or qualifying native measurements. |
| 2026-10-15 | Created in the plan phase. |
| 2026-10-15 | Review remediation: per-model milestones so the GLM-5.3 closure does not wait for DeepSeek-V4-Flash work; the DeepSeek-V4-Flash cost model waits for 09 `full_frontier_path[GLM]` (S1, E4, Q12, Q13). |

**What to build:** Frontier prices TP collectives, MoE dispatch and combine, and NIXL KV transfers from rerunnable native measurements. The measurements use the libraries the G0-B runs actually resolved, at the world sizes, payloads, and concurrency the target topology uses, with stated validity ranges and holdout errors. The ticket also produces evidence of whether KV transfer and EP traffic contend for the same NIC. (Spec v2 T06, §6.)

**Blocked by:**
- `raw_measurements[DSV4]`: matching05 structural path, actual-hardware01
  qualification,03 DSV4 cache inventory and17 case shapes. Reuse valid measured
  rows first; preserve native sample/owner/transport scope. H800 is not H200.
- `narrow_cost_model[GLM]`: `raw_measurements[GLM]`, and `12[GLM]` (D-PD) for any shared-resource or concurrency abstraction.
- `narrow_cost_model[DSV4]`: `raw_measurements[DSV4]`, the existing shared measured backend, and `12[DSV4]`; no blanket dependency on GLM's full-path or calibration completion under D-CALIBRATION-PRIORITIES.

Active06[DSV4] closes when its raw measurements and narrow cost model close.
GLM raw/model milestones are PAUSED_PENDING_USER_APPROVAL,not a current edge.

**Status:** PARTIAL. P0 binds216 retained EP16 rows plus54 exact-supplement rows through `data/profiling/network/h800_dgx/glm_p0_ep16_20261021b/summary.csv` (1241 rows; prior1187 unchanged). Exact SP768/1,75,595,776B job `exp-1021-214338-130275` succeeded on16 non-preemptible H800: both containers exit0/no restart,0.882222GPUh.54 timing rows,248 traces/30,296,104B and160 native Reduce calls per rank pass. Actual P/D lookup returns0.731449604ms; SP769/1 stays typed-rejected. See [primary report](../../task_2026-10-21_p0_input_binding/test_report_2026-10-21_p0_exact_reduce_primary.md). Complete NIXL/overlap/formal coverage remains open.


## Current Codex checkpoint — 2026-10-31

Use [current plan](../plan.md#current-execution-route--codex-cli-h200-operator-path-2026-10-31) and [current handoff](../progress.md#current-session-handoff--codex-cli-2026-10-31). D-H200-MAINLINE-CONTINUE and decision 46 apply; Codex uses personal StepMind Python RJobBackend directly. GLM stays paused. Dated snapshots below are historical where they conflict with this entry.

Measured communication remains the formal input. B29 A8: P2-32 6,310,644 and P2-64 9,002,176 calls, all exact, zero violations. B30b owns diagnostic F1b link and F3 H200/V4 NIXL probes/fits. Collectives-only03 completed Succeeded: 1,644 measured rows and 548 kernel records verify against the plan. Its positive link fit fails G5h validation; both prescribed F1b CPU replays failed compute coverage (embedding descriptor32; query16 in padded24 attention group), so complete E2E readings are unavailable. See [F1b report](../test_report_2026-11-01_f1b_h200_links.md). NIXL02 completed all 240 records with matching notifications and CSV reconstruction; its affine fit fails (negative intercept, p90 residual 29.18%). No replacement F3 fields or replay exist. See [F3 report](../test_report_2026-11-01_f3_h200_nixl.md). No cross-hardware fit or diagnostic input is promoted automatically.

## Historical priority-case workload coverage — 2026-10-21

Scope override2026-10-22:D-DSV4-ONLY-GLM-PAUSED. Current simulation/tests/
calibration/new packages cover DSV4 only;all GLM tiers and measurement-proposal
follow-up are parked pending new explicit user approval. Keep prior code/data/
receipts and UNVERIFIED metrics. Reuse the accepted measured backend and matching
216 EP16/54 Reduce rows where actual V4 scope matches;legacy glm input/helper
names do not change model identity. No new GLM consumer/test/completion is needed.

Derive collective and request-level NIXL payload ranges from the 6144-token
unchunked prompt, reached decode contexts through total length 7168 and canonical
per-role TP8/DP2/EP16/PP1 for32 GPUs or TP8/DP4/EP32/PP1 for64 GPUs, with128
requests at QPS1. Prioritize P0 H80032 and reuse matching H800 curves. The
measured backend remains formal: every reached shape on the actual hardware must
resolve from a matching curve or return a typed coverage error, never extrapolate
or use another hardware's curve. The127s arrival span is unrelated to microbenchmark timing
or the formal THPT denominator. Actual graph descriptors, lane concurrency and
memory/KV settings come from native evidence; this update does not freeze them.

H200 transport/UCX-selection admission remains pending. Its historical70/130-cap
packages are not submittable; any new generation follows personal H200
non-preemptible<=32 and otherwise preemptible. This does not block independent
P0 native reference work or waive NIXL transport evidence.

## Historical mainline recovery — 2026-10-20

G0-B[GLM] closed on 2026-10-20, so `raw_measurements[GLM]` waits only for `03.cache_inventory[GLM]` (plan workstream W5). Measure on H800 at the target groups: TP8 attention collectives, sequence-parallel MoE all-gather/reduce-scatter at EP16/EP32, and NIXL READ for the GLM page layout. Observer qualification is not required for these primitive measurements.

### Status at the 2026-10-20 handoff

- **Measurement code:** `45545fe`, device-time fix `8b63827`, device-aligned starts plus shuffled passes plus EP-under-NIXL stage `1f17199`.
- **H800 runs (codesign, all Succeeded):** smoke_02 exp-1020-101140-754756 (PASS 5/5); first campaign run concurrently — TP8/DP2/EP16 exp-1020-102139-753975, TP8/DP4/EP32 exp-1020-102139-615068, NIXL READ exp-1020-102139-729443 (curves provisional: run-to-run spread from one node's start skew). K5 run one after another, 3 passes x 30 repeats each — DP4/EP32 exp-1020-134023-656882, DP2/EP16 exp-1020-135432-898628, NIXL READ exp-1020-140146-211735, EP+NIXL overlap exp-1020-142252-986931.
- **K5 results** ([w5_glm_comm_20261020/report.md](../w5_glm_comm_20261020/report.md) section 3.6): DP2-vs-DP4 same-TP spread median 3.4% / p90 11.9%; leave-one-out median dispatch 5%, combine 2%, all-reduce 7%, reduce-scatter 6%; all-gather 19% mixed (logits rows 9-29% slower at equal bytes; about 5% each once split). EP is 7-38% slower when NIXL READ shares the NIC (every point above budget; contention modelling belongs to ticket 12). NIXL READ fit 126.2 Gbit/s + 0.0647 ms.
- **Curated curves:** `/data/ycfeng/tmp/glm53_dsv4/w5_comm_k5/summary.csv` (971 rows) and `nixl_summary.csv` (186 rows); to be checked into `data/profiling/network/h800_dgx/` with K4 (D-CURVE-HOME).
- **Open:** K1-K4 implementation (approved, D-W7-COMM-APPROVED; [design card](../w7_glm_comm_model_20261020/design.md)); logits pricing (recommended option B: drop logits rows from the all-gather curve and price logits with the hidden-state curve, about 0.1% of a step); uneven-lane NCCL kernel switch (combine up to -85%); EP32 measured on one node set only; prefill above 8,192 tokens per DP rank is outside coverage. Acceptance items "Real EP32 and cross-node NIXL" are measured for GLM only; DSV4 is not measured.

## Existing-consumer preparation — 2026-10-18

`../ticket06_curve_preparation.md` characterizes the current Vidur/analytical collective and KV-transfer consumers. Six scoped source pins and three existing P2P CPU tests passed; synthetic tests are not native EP/NIXL, wire or same-NIC evidence. Reviewer73a5c5c2 completed one P2: the Vidur fallback is affected by `num_devices`, not `comm_domain`, because the call/signature do not accept the latter. Primary corrected that sentence only and verified the exact AST/signatures; no post-correction independent No findings verdict is claimed. Source/report/receipt/test evidence and the original verdict are separately retained in `../parallel_preparation_20261018/concurrent_wave_20261018/manifest.json`76entries and `completion_20261018/manifest.json`56entries. See `../test_report_2026-10-18_concurrent_preparation.md`. No native measurements, new curves, coverage acceptance, consumer implementation or D-PD decision; both formal milestones remain blocked.

## Milestones

### raw_measurements (active DSV4 only; GLM retained paused)
- [ ] Reuse or measure touched collective/NIXL curves on each priority case's actual hardware/groups/payloads, H80032 first. H800 evidence does not qualify H200.
- [x] Microbenchmarks on the resolved EP backend and NIXL API (`nccl-tests` only for hardware diagnosis): world sizes 8/16/32 as used, payload quantiles and boundaries from that model's cache inventory and routing samples, and valid concurrency levels among 1/2/4/8/16. Prefill throughput and decode latency paths are kept separate. A curve already measured for GLM-5.3 is reused for DeepSeek-V4-Flash only when the library, world size, payload range, and concurrency match.
- [x] Record per-rank distributions and critical-path maxima.
- [ ] Assign each fused MoE either a measured composite scope or observable segments, so that no time is charged both inside and outside the fused kernel.
- [x] Run an EP+NIXL same-NIC overlap experiment (evidence for `12[m]` contention conclusion). Lifecycle disposition in 12 precedes this measurement; it does not wait for the entire curve campaign.

### narrow_cost_model (active DSV4 only; GLM retained paused)
- [ ] Bind qualified DSV4 curves to the existing CC/KV-transfer consumers with
  explicit source/range ownership. No new GLM consumer/completion dependency;
  retain typed misses and seek approval for any material new core abstraction.
- [ ] Report the input range, holdout, error, and coverage limits per curve. Out-of-range inputs return a coverage status.

## Acceptance

- [ ] Real EP32 and cross-node NIXL are covered for active DSV4 cases;paused GLM
  completion is not required and missing V4 evidence is not waived.
- [ ] No overlapping charges.
- [ ] No bandwidth or latency constant without a source.
- [ ] No model, QPS, or GPU-count compensation branches.
