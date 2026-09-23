# Workflow-gap summary — stage_admission_case_001

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-23 | R-10: rerun against `after-r2` with the negative controls as rows N1/N4; dense V5 cause restated with both sources and the derived barrier-aligned M5. |
| 2026-09-23 | D-9 adopted: dense V5 reported, not gated; rerun status PASS, 0 mismatches. |
| 2026-09-23 | Created from run `sa-pp-20260923b` against Frontier sets `base` (`1f694f7` rule) and `after` (`dac4e69`). |

## Inputs

| Side | Source |
| --- | --- |
| vLLM | `runs/vllm-instrumented/sa-pp-20260923b/` — vLLM-BS `494b9f327` plus `inputs/groundtruth_overlay.patch` (SHA-256 `8d476789…3a9c81`), DP=2, PP=2, TP=1, MoE with EP, 4×H800 |
| Frontier before | `/data/ycfeng/tmp/stage_admission_ordering/base/G7-*` |
| Frontier after | `/data/ycfeng/tmp/stage_admission_ordering/after-r2/G7-*` (R-10; byte-identical to set `after` at `dac4e69`) |
| Producer | `tests/comparison/stage_admission_pp/compare_lanes.py` → `workflow_gap_table.csv`, `lane_metrics.json`, `workflow_gap_status.json` |

## Result

56 rows: 50 `MATCH`, 0 `MISMATCH`, 2 `INFORMATIONAL` (dense V5 under D-9),
4 `HOLDS` (negative controls); status PASS, `negative_control_holds = true`.
`vllm_placement_ok = true`, no unseen request. Before R-10 the controls were
folded into V1 and dense V4 (52 rows).

| Metric | MoE (n8, n16) | Dense (n8, n16) |
| --- | --- | --- |
| V1 completion | MATCH in 6/6 rounds | MATCH in 6/6 rounds |
| N1 / N4 base control | N1 HOLDS (n8, n16): base `admission_deadlock` | N4 HOLDS (n8, n16): base co-start 1.0 |
| V2 lane sequences | MATCH 6/6 | MATCH 6/6 |
| V3 stage-0 pairing | MATCH 6/6 | MATCH 6/6; base pairs are shifted by one forward |
| V4 first-forward co-start | MATCH 6/6 (vLLM ≤ 0.063, after 0.0) | MATCH 6/6 (vLLM ≤ 0.248, after 0.0) |
| V5 stage-0 co-execution | MATCH: vLLM 0.976 / 0.948, after 1.0 | INFORMATIONAL (D-9): vLLM 0.706 / 0.865, after 1.0, base 0.600 / 0.778 |

## Dense V5 (MISMATCH before D-9)

- Observed: vLLM's dense co-execution varies from round to round by more than
  the 0.10 bound: 0.537–0.926 over the 12 dense rounds of runs a and b. The
  n16 means of the two runs differ by 0.18.
- Observed (`co_execution_decomposition_sa-pp-20260923{a,b}.json`): the dense
  non-overlap has two sources. Every pair overlaps, so it splits exactly.
  - Per-pair start offsets. `forward_start_ts` is taken before the
    per-forward DP metadata all-reduce, so the rank that arrives first
    records its wait as busy time. Start offsets exceed end offsets in 3 of
    the 6 dense rounds of run b.
  - End offsets from per-rank duration variation: CV 0.10–0.29 on forwards of
    about 3 ms.
  MoE ends stay within 0.2–0.7 ms in total.
- Derived, not measured (R-10): with both starts of each pair set to the
  later one, where the all-reduce releases both ranks, M5 is 0.66–0.98 for
  dense and 0.988–0.994 for MoE, higher than observed in every round. The
  traces carry no timestamp after the exchange.
- Inference (restated at R-10): neither source is an admission difference.
  Both ranks enter the same forward, which V1–V4 measure, and they match in
  every round. The Frontier owner of the dense gap is the execution-time
  model: the dummy predictor models neither the pre-exchange wait nor the
  duration variation. It is not `stage_execution_context.py`, the default
  owner label written into the table. The first version of this inference
  named only the duration variation.
- The first analysis stopped here with nothing adjusted. The user adopted
  D-9: V5 is informational for the dense shape and stays a gate for MoE.
  C7 rests on V1–V4 for both models, V5 for MoE, and the base negative
  controls.
